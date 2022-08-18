#!/bin/bash

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

echo '
This script installs the latest offical pandoc binary for Linux. It is primarily
meant for the CI/CD job. It must run from the docs/ directory.

Advance users might directly use this script on their Linux box, however be
aware of the following disclaimer:

THE SCRIPT IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'

ARCH=""
case $(uname -m) in
    x86_64|amd64)
        ARCH=amd64
        ;;
    aarch64|arm64)
        ARCH=arm64
        ;;
    *)
        echo "Unknown architecture: $(uname -m)" >&2
        exit -1
esac

extract_pandoc_version() {
    local version
    local download_url

    # https://github.com/jgm/pandoc/releases/download/2.18/pandoc-2.18-linux-amd64.tar.gz
    download_url=$1

    # 2.18-linux-amd64.tar.gz
    version=${download_url#*pandoc-}

    # 2.18
    version=${version%*-linux-*.tar.gz}

    echo $version
}

latest_pandoc_builds() {
  curl --silent "https://api.github.com/repos/jgm/pandoc/releases/latest" | # Get latest release from GitHub api
    grep '"browser_download_url":' |                                # Get download url
    sed -E 's/.*"([^"]+)".*/\1/'                                    # Pluck JSON value
}

# https://github.com/jgm/pandoc/releases/download/2.18/pandoc-2.18-linux-amd64.tar.gz
TARBALL=$(latest_pandoc_builds | grep linux-${ARCH}.tar.gz)
echo $TARBALL

# 2.18
VERSION=$(extract_pandoc_version $TARBALL)
echo $VERSION

mkdir -p _build/bin
curl -L $TARBALL | tar -C _build/ -xzvf - pandoc-$(extract_pandoc_version $TARBALL)/bin/pandoc
[[ -L _build/bin/pandoc ]] && rm _build/bin/pandoc
ln -s ../pandoc-$VERSION/bin/pandoc _build/bin/pandoc
