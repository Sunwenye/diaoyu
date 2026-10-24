#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量上传本地目录到 GitHub 仓库（无需安装 git，直接走 GitHub REST API）。

为什么用 API 而不是 git：
    有些机器没装 git，或者不想为一个仓库初始化本地版本库。
    GitHub 的 Git Data API 允许直接创建 blob -> tree -> commit -> ref，
    一次提交把整个目录推上去，效果和 git push 一致。

关于空仓库的特殊处理（重要）：
    刚创建、还没有任何提交的仓库，调用 git/blobs 会返回
    HTTP 409 {"message":"Git Repository is empty."}
    这是 GitHub 的限制，不是权限问题。
    解决办法：先用 Contents API（PUT /contents/路径）创建第一个提交，
    仓库有了 HEAD 之后，Git Data API 就能正常使用了。
    本脚本已自动处理这种情况。

安全说明：
    * Token 只从环境变量 GH_TOKEN 读取，绝不硬编码进代码。
      硬编码的 token 一旦提交到公开仓库，会永久留在 Git 历史里，
      即使之后删掉文件也仍然可被翻出来，爬虫几分钟内就能扫到。
    * 上传前会扫描每个待传文件的正文，发现疑似明文 token 立即中止。
    * 默认尊重 .gitignore 的常见忽略模式（__pycache__、*.pyc、venv 等）。

环境变量：
    GH_TOKEN    必填，GitHub Personal Access Token（需 repo 权限）
    GH_BRANCH   选填，目标分支，默认 main

用法：
    # Windows PowerShell
    $env:GH_TOKEN = "ghp_你的token"
    python upload_to_github.py --repo 用户名/仓库名 --dir E:\\diaoyu
    Remove-Item Env:\\GH_TOKEN          # 用完清掉

    # bash / macOS
    GH_TOKEN=ghp_你的token python upload_to_github.py --repo 用户名/仓库名 --dir ./

    # 先看看会传哪些文件，不实际上传
    python upload_to_github.py --repo 用户名/仓库名 --dir . --dry-run

依赖： requests     (pip install requests)
"""

import argparse
import base64
import fnmatch
import os
import re
import sys

try:
    import requests
except ImportError:
    print("缺少依赖，请先执行：pip install requests")
    sys.exit(2)

API = "https://api.github.com"

# 默认忽略模式，与常见 .gitignore 保持一致
DEFAULT_IGNORES = [
    "__pycache__/", "*.pyc", "*.pyo", "*.pyd",
    ".git/", "venv/", ".venv/", "env/", "ENV/",
    "build/", "dist/", "downloads/", "eggs/", ".eggs/",
    "*.egg-info/", "*.egg", "sdist/", "wheels/", "var/",
    ".idea/", ".vscode/", ".mypy_cache/", ".pytest_cache/",
    ".ipynb_checkpoints/", "target/", "htmlcov/",
    ".DS_Store", "Thumbs.db", "*.swp", "*.swo", "*~",
    ".coverage", "coverage.xml", "pip-log.txt",
]

# 疑似凭据特征：任一命中就拒绝上传该文件
SECRET_PATTERNS = [
    re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bgh[ousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
]

# 永远不上传的文件名（即使没匹配到上面的模式）
NEVER_UPLOAD_NAMES = {
    ".env", ".env.local", ".env.production", ".netrc", ".npmrc",
    ".pypirc", "id_rsa", "id_ed25519", "id_dsa",
}


def log(msg):
    print(msg, flush=True)


def b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def scan_secrets(text):
    """返回命中的凭据特征名列表；空列表表示干净。"""
    hits = []
    for pat in SECRET_PATTERNS:
        if pat.search(text):
            hits.append(pat.pattern[:40])
    return hits


def is_ignored(rel_path, ignores):
    """判断相对路径是否命中忽略规则。"""
    norm = rel_path.replace(os.sep, "/")
    parts = norm.split("/")
    for pat in ignores:
        if pat.endswith("/"):
            dirname = pat[:-1]
            if any(p == dirname for p in parts[:-1]):
                return True
            continue
        base = parts[-1]
        if fnmatch.fnmatch(base, pat) or fnmatch.fnmatch(norm, pat):
            return True
    return False


def collect_files(root, ignores, extra_exclude=()):
    """递归收集待上传文件，返回 (相对路径, 绝对路径) 列表。"""
    result = []
    excluded = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root).replace(os.sep, "/")
        if rel_dir == ".":
            rel_dir = ""
        # 就地裁剪子树，避免遍历被忽略的目录
        dirnames[:] = [
            d for d in dirnames
            if not is_ignored((rel_dir + "/" + d).lstrip("/"), ignores)
        ]
        for fn in filenames:
            abs_path = os.path.join(dirpath, fn)
            rel_path = os.path.relpath(abs_path, root).replace(os.sep, "/")
            if is_ignored(rel_path, ignores):
                excluded.append((rel_path, "ignore-rule"))
                continue
            if fn in NEVER_UPLOAD_NAMES:
                excluded.append((rel_path, "sensitive-filename"))
                continue
            if rel_path in extra_exclude:
                excluded.append((rel_path, "explicit-exclude"))
                continue
            result.append((rel_path, abs_path))
    result.sort(key=lambda x: x[0])
    return result, excluded


def preflight_scan(files):
    """上传前的安全预检：文本文件扫描凭据，二进制文件仅检查大小。"""
    log("\n[preflight] 扫描待上传文件是否含明文凭据 ...")
    blocked = []
    for rel, abs_path in files:
        size = os.path.getsize(abs_path)
        if size > 95 * 1024 * 1024:
            blocked.append((rel, "文件超过 95MB，API 单次上传上限约 100MB"))
            continue
        try:
            with open(abs_path, "rb") as f:
                raw = f.read()
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue  # 二进制文件，跳过正则扫描
        except OSError as e:
            blocked.append((rel, "读取失败: %s" % e))
            continue
        hits = scan_secrets(text)
        if hits:
            blocked.append((rel, "检测到疑似凭据: %s" % ", ".join(hits)))

    if blocked:
        log("\n" + "=" * 62)
        log("中止上传：以下文件存在风险，未推送任何内容")
        log("=" * 62)
        for rel, why in blocked:
            log("  BLOCKED  %-38s %s" % (rel, why))
        log("\n处理办法：把凭据改成从环境变量读取，或将该文件加入排除列表。")
        return False
    log("    clean，%d 个文件可安全上传" % len(files))
    return True


def get_head_sha(headers, repo, branch):
    """返回分支 HEAD 的 commit sha；仓库为空时返回 None。"""
    r = requests.get(API + "/repos/%s/git/refs/heads/%s" % (repo, branch),
                     headers=headers, timeout=45)
    if r.status_code == 200:
        return r.json()["object"]["sha"]
    return None


def bootstrap_empty_repo(headers, repo, branch, rel, abs_path, author, message):
    """
    空仓库无法用 git/blobs（会 409 "Git Repository is empty."）。
    改用 Contents API 创建第一个文件，从而产生仓库的首个提交。
    """
    log("\n[bootstrap] 仓库为空，用 Contents API 创建首个提交 ...")
    log("    file: %s" % rel)
    r = requests.put(API + "/repos/%s/contents/%s" % (repo, rel),
                     headers=headers, timeout=180,
                     json={
                         "message": message,
                         "content": b64(abs_path),
                         "branch": branch,
                         "author": author,
                         "committer": author,
                     })
    if r.status_code not in (200, 201):
        log("    FAIL -> HTTP %d %s" % (r.status_code, r.text[:300]))
        return None
    sha = (r.json().get("commit") or {}).get("sha", "")
    log("    首个提交已创建: %s" % sha[:12])
    return sha


def main():
    ap = argparse.ArgumentParser(
        description="批量上传本地目录到 GitHub 仓库（REST API，无需 git）")
    ap.add_argument("--repo", required=True, help="形如 用户名/仓库名")
    ap.add_argument("--dir", default=".", help="要上传的本地目录，默认当前目录")
    ap.add_argument("--branch", default=os.environ.get("GH_BRANCH", "main"),
                    help="目标分支，默认 main")
    ap.add_argument("--message", default=None, help="提交信息")
    ap.add_argument("--dry-run", action="store_true",
                    help="只列出将要上传的文件，不实际推送")
    ap.add_argument("--exclude", action="append", default=[],
                    help="额外排除的相对路径，可重复指定")
    ap.add_argument("--allow-secret-file", action="append", default=[],
                    help="⚠ 明知含凭据仍要上传的文件（强烈不建议）")
    args = ap.parse_args()

    token = os.environ.get("GH_TOKEN", "").strip()
    if not token:
        log("ERROR: 未设置环境变量 GH_TOKEN")
        log("  PowerShell :  $env:GH_TOKEN = \"ghp_你的token\"")
        log("  bash       :  GH_TOKEN=ghp_你的token python upload_to_github.py ...")
        sys.exit(2)

    src = os.path.abspath(args.dir)
    if not os.path.isdir(src):
        log("ERROR: 目录不存在: %s" % src)
        sys.exit(2)

    headers = {
        "Authorization": "token " + token,
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "upload_to_github.py",
    }

    # ---------- 0. 验证 token 与仓库 ----------
    log("[0] 验证 token 与仓库 %s ..." % args.repo)
    r = requests.get(API + "/user", headers=headers, timeout=45)
    if r.status_code != 200:
        log("    token 无效或已过期 (HTTP %d)" % r.status_code)
        sys.exit(1)
    login = r.json().get("login")
    uid = r.json().get("id", "")
    scopes = r.headers.get("X-OAuth-Scopes", "")
    log("    身份: %s   scopes: %s" % (login, scopes or "(none)"))
    if "repo" not in scopes:
        log("    WARNING: token 缺少 repo 权限，推送可能失败")

    r = requests.get(API + "/repos/" + args.repo, headers=headers, timeout=45)
    if r.status_code != 200:
        log("    仓库不可访问 (HTTP %d)" % r.status_code)
        sys.exit(1)
    info = r.json()
    log("    仓库: %s   可见性: %s   默认分支: %s"
        % (info.get("full_name"),
           "private" if info.get("private") else "public",
           info.get("default_branch")))
    if not (info.get("permissions") or {}).get("push"):
        log("    ERROR: 当前 token 对该仓库没有 push 权限")
        sys.exit(1)

    # 用 GitHub noreply 邮箱，避免在公开提交里暴露真实邮箱
    author = {"name": login,
              "email": "%s+%s@users.noreply.github.com" % (uid, login)}

    # ---------- 1. 收集文件 ----------
    log("\n[1] 扫描目录 %s ..." % src)
    files, excluded = collect_files(src, DEFAULT_IGNORES, tuple(args.exclude))
    for rel, why in excluded:
        log("    skip %-40s (%s)" % (rel, why))
    total = sum(os.path.getsize(a) for _, a in files)
    log("    待上传 %d 个文件，共 %.1f KB" % (len(files), total / 1024.0))
    for rel, abs_path in files:
        log("      + %-42s %8d bytes" % (rel, os.path.getsize(abs_path)))

    if not files:
        log("\n没有可上传的文件，退出。")
        sys.exit(0)

    # ---------- 2. 安全预检 ----------
    if args.allow_secret_file:
        log("\n[!] 已用 --allow-secret-file 显式放行: %s"
            % ", ".join(args.allow_secret_file))
        files = [(r, a) for r, a in files if r not in args.allow_secret_file]
    if not preflight_scan(files):
        sys.exit(3)

    if args.dry_run:
        log("\n[dry-run] 未推送任何内容。去掉 --dry-run 才会真正上传。")
        return

    message = args.message or ("Add %d files from %s"
                               % (len(files), os.path.basename(src)))

    # ---------- 3. 判断是否空仓库，必要时先引导 ----------
    head = get_head_sha(headers, args.repo, args.branch)
    remaining = files
    bootstrapped = False

    if head is None:
        log("\n[3] 分支 %s 不存在 —— 仓库没有任何提交（空仓库）" % args.branch)
        # 优先用 README.md 作为首个提交，最自然；否则用排序后的第一个文件
        pick = None
        for rel, abs_path in files:
            if rel.lower() in ("readme.md", "readme.rst", "readme.txt"):
                pick = (rel, abs_path)
                break
        if pick is None:
            pick = files[0]
        sha = bootstrap_empty_repo(headers, args.repo, args.branch,
                                   pick[0], pick[1], author,
                                   "Initial commit: add %s" % pick[0])
        if not sha:
            log("引导提交失败，退出。")
            sys.exit(1)
        head = sha
        bootstrapped = True
        remaining = [(r, a) for r, a in files if r != pick[0]]
        log("    剩余 %d 个文件将合并为第二个提交" % len(remaining))
    else:
        log("\n[3] 分支 %s 已存在，HEAD = %s，将追加一个提交"
            % (args.branch, head[:12]))

    if not remaining:
        log("\n所有文件已通过引导提交完成，无需第二个提交。")
        final_sha = head
    else:
        # ---------- 4. 创建 blobs ----------
        log("\n[4] 上传文件内容 (git blobs) ...")
        tree_entries = []
        for rel, abs_path in remaining:
            r = requests.post(API + "/repos/%s/git/blobs" % args.repo,
                              headers=headers, timeout=180,
                              json={"content": b64(abs_path), "encoding": "base64"})
            if r.status_code not in (200, 201):
                log("    FAIL %s -> HTTP %d %s" % (rel, r.status_code, r.text[:200]))
                sys.exit(1)
            sha = r.json()["sha"]
            tree_entries.append({"path": rel, "mode": "100644",
                                 "type": "blob", "sha": sha})
            log("    blob ok  %-42s %s" % (rel, sha[:10]))

        # ---------- 5. 取 base tree 并创建新 tree ----------
        r = requests.get(API + "/repos/%s/git/commits/%s" % (args.repo, head),
                         headers=headers, timeout=45)
        if r.status_code != 200:
            log("    FAIL 读取 HEAD commit -> HTTP %d" % r.status_code)
            sys.exit(1)
        base_tree = r.json()["tree"]["sha"]

        log("\n[5] 创建 tree (base_tree=%s) ..." % base_tree[:10])
        r = requests.post(API + "/repos/%s/git/trees" % args.repo,
                          headers=headers, timeout=120,
                          json={"base_tree": base_tree, "tree": tree_entries})
        if r.status_code not in (200, 201):
            log("    FAIL tree -> HTTP %d %s" % (r.status_code, r.text[:300]))
            sys.exit(1)
        tree_sha = r.json()["sha"]
        log("    tree sha: %s" % tree_sha)

        # ---------- 6. 创建 commit ----------
        log("\n[6] 创建 commit ...")
        msg = message if not bootstrapped else (message + " (remaining files)")
        r = requests.post(API + "/repos/%s/git/commits" % args.repo,
                          headers=headers, timeout=120,
                          json={"message": msg, "tree": tree_sha,
                                "author": author, "committer": author,
                                "parents": [head]})
        if r.status_code not in (200, 201):
            log("    FAIL commit -> HTTP %d %s" % (r.status_code, r.text[:300]))
            sys.exit(1)
        final_sha = r.json()["sha"]
        log("    commit sha: %s" % final_sha)

        # ---------- 7. 更新 ref ----------
        log("\n[7] 更新 refs/heads/%s ..." % args.branch)
        r = requests.patch(API + "/repos/%s/git/refs/heads/%s" % (args.repo, args.branch),
                           headers=headers, timeout=120,
                           json={"sha": final_sha, "force": False})
        if r.status_code != 200:
            log("    FAIL ref -> HTTP %d %s" % (r.status_code, r.text[:300]))
            log("    blobs/tree/commit 已创建但分支指针未更新，可重试本脚本。")
            sys.exit(1)
        log("    ref ok")

    # ---------- 8. 回读验证 ----------
    log("\n[8] 回读远程仓库验证 ...")
    r = requests.get(API + "/repos/%s/git/trees/%s?recursive=1"
                     % (args.repo, args.branch), headers=headers, timeout=60)
    remote = {}
    if r.status_code == 200:
        for it in r.json().get("tree", []):
            if it.get("type") == "blob":
                remote[it["path"]] = it.get("size")
    ok, missing = 0, []
    for rel, abs_path in files:
        local_size = os.path.getsize(abs_path)
        if rel in remote:
            if remote[rel] == local_size:
                ok += 1
                log("    MATCH  %-42s %d bytes" % (rel, remote[rel]))
            else:
                missing.append("%s (size %s != local %d)"
                               % (rel, remote[rel], local_size))
        else:
            missing.append("%s (not found remotely)" % rel)

    log("\n" + "=" * 62)
    log("上传结束")
    log("  仓库 : https://github.com/%s" % args.repo)
    log("  分支 : %s" % args.branch)
    log("  提交 : %s" % final_sha)
    log("  提交数: %s" % ("2 (含空仓库引导提交)" if bootstrapped and remaining else "1"))
    log("  验证 : %d / %d 个文件字节数完全一致" % (ok, len(files)))
    if missing:
        log("  ⚠ 异常 :")
        for m in missing:
            log("      - %s" % m)
    log("=" * 62)
    log("\n安全提醒：如果这个 token 曾出现在聊天记录、脚本或日志里，")
    log("请到 GitHub Settings -> Developer settings -> Personal access tokens")
    log("立即撤销并重新生成。")

    sys.exit(0 if not missing else 4)


if __name__ == "__main__":
    main()
