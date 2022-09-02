# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
import os
from typing import List

from setuptools import find_namespace_packages, setup

_repo: str = "amazon-accessible-rl-sdk"
_pkg: str = "a2rl"
_version = "1.0.1-dev"


def read_lines(fname: str) -> List[str]:
    """Read the content of a file.

    You may use this to get the content of, for e.g., requirements.txt, VERSION, etc.
    """
    return open(os.path.join(os.path.dirname(__file__), fname)).readlines()


def read_requirements(fname: str) -> List[str]:
    lines = [
        flip_git(stripped_line)
        for stripped_line in (line.strip() for line in read_lines(fname))
        if stripped_line != ""
    ]

    return lines


def flip_git(s: str) -> str:
    """Flip pip's git source from ``requirements.txt`` format to `setuptools` format.

    If `s` is not a git source, return as-is.

    Args:
        s (str): a line in ``requirements.txt``.

    Returns:
        str: if `s` is ``git+.://gh.com/a/b@c#egg=d``, then return ``d @ git+.://gh.com/a/b@c``.
        Otherwise, return as-is.
    """
    if not s.startswith("git+"):
        return s
    git_url, egg = s.rsplit("#", 1)
    _, egg_name = egg.split("=")
    return f"{egg_name} @ {git_url}"


# Declare minimal set for installation
required_packages: List[str] = read_requirements("requirements.txt")
extras = {
    "dev": read_requirements("requirements-dev.txt"),
    "test": read_requirements("requirements-test.txt"),
}

setup(
    name=_pkg,
    packages=find_namespace_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    version=_version,
    description="Make recommendations for sequential decision problems using offline data",
    long_description="".join(read_lines("README.md")),
    long_description_content_type="text/markdown",
    author="AWS/ProServe Global Team",
    url=f"https://github.com/awslabs/{_repo}/",
    download_url="",
    project_urls={
        "Bug Tracker": f"https://github.com/awslabs/{_repo}/issues/",
        "Documentation": f"https://{_repo}.readthedocs.io/en/stable/",
        "Source Code": f"https://github.com/awslabs/{_repo}/",
    },
    license="Apache License 2.0",
    platforms=["any"],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8.0",
    install_requires=required_packages,
    extras_require=extras,
)
