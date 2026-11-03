#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Select a compatible vendor wheel from one HCU simple index, without installing."""

import argparse
from html.parser import HTMLParser
import sys
from urllib.error import URLError
from urllib.parse import unquote, urljoin, urlsplit
from urllib.request import Request, urlopen

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.tags import sys_tags
from packaging.utils import InvalidWheelFilename, canonicalize_name, parse_wheel_filename


class PackageLinks(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            attrs = dict(attrs)
            if attrs.get("href") and "data-yanked" not in attrs:
                self.links.append(attrs)


def select_wheel(html, index_url, package, dtk_ver, torch_version,
                 compatible_tags=None, python_version=None):
    parser = PackageLinks()
    parser.feed(html)
    tag_ranks = {tag: rank for rank, tag in enumerate(
        sys_tags() if compatible_tags is None else compatible_tags
    )}
    python_version = python_version or ".".join(map(str, sys.version_info[:3]))
    expected = {f"dtk{dtk_ver}", "torch" + torch_version.replace(".", "")}
    candidates = []
    available = set()
    for attrs in parser.links:
        wheel_url = urljoin(index_url, attrs["href"])
        parts = urlsplit(wheel_url)
        # Only use files served by the requested HCU index host.
        if parts.scheme not in ("http", "https") or parts.netloc != urlsplit(index_url).netloc:
            continue
        filename = unquote(parts.path.rsplit("/", 1)[-1])
        try:
            name, version, build, tags = parse_wheel_filename(filename)
        except InvalidWheelFilename:
            continue
        if name != canonicalize_name(package):
            continue
        available.add(str(version))
        if version.is_prerelease or not expected.issubset(set((version.local or "").split("."))):
            continue
        ranks = [tag_ranks[tag] for tag in tags if tag in tag_ranks]
        if not ranks:
            continue
        try:
            if not SpecifierSet(attrs.get("data-requires-python") or "").contains(python_version):
                continue
        except InvalidSpecifier:
            continue
        candidates.append((version, build, -min(ranks), wheel_url))
    if not candidates:
        versions = ", ".join(sorted(available)) or "none"
        raise ValueError(
            f"No compatible HCU {package} wheel for dtk{dtk_ver}, torch{torch_version}, "
            f"Python {python_version} and this platform in {index_url}. "
            f"Available versions: {versions}. Public-package fallback is disabled."
        )
    version, _, _, wheel_url = max(candidates)
    return str(version), wheel_url


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-url", required=True)
    parser.add_argument("--package", required=True)
    parser.add_argument("--dtk-ver", required=True)
    parser.add_argument("--torch-version", required=True)
    args = parser.parse_args()
    index_url = args.index_url.rstrip("/") + "/" + canonicalize_name(args.package) + "/"
    request = Request(index_url, headers={"Accept": "application/vnd.pypi.simple.v1+html"})
    try:
        with urlopen(request, timeout=30) as response:
            html = response.read().decode("utf-8")
        version, wheel_url = select_wheel(
            html, index_url, args.package, args.dtk_ver, args.torch_version
        )
    except (OSError, URLError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Selected HCU {args.package}: {version}", file=sys.stderr)
    # stdout is consumed by setup-env.sh as one quoted pip requirement URL.
    print(wheel_url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
