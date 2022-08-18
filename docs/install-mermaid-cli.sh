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
This script installs the latest offical mermaid-cli. It is primarily meant for
the CI/CD job. It must run from the docs/ directory.

Advance users might directly use this script on their Linux box, however be
aware of the following disclaimer:

THE SCRIPT IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'


export NVM_DIR=$HOME/.nvm
mkdir -p $NVM_DIR

# Install nvm to NVM_DIR
echo "Installing nvm..."
NVM_VERSION=$(curl -sL https://api.github.com/repos/nvm-sh/nvm/releases/latest | jq -r '.name')
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/${NVM_VERSION}/install.sh | bash
[[ -s "$NVM_DIR/nvm.sh" ]] && . "$NVM_DIR/nvm.sh"  # Loads nvm
# [[ -s "$NVM_DIR/bash_completion" ]] && . "$NVM_DIR/bash_completion"  # Loads nvm bash_completion
echo "Checking nvm:" `nvm --version`

# Install node.js (use lts version as-per CDK recommendation)
echo "Installing node.js and npm..."
nvm install --lts
nvm use --lts
npm install -g npm
node -e "console.log('Running Node.js ' + process.version)"
echo "Checking npm:" `npm -v`

# Install mermaid-cli
echo "Installing mermaid-cli..."
npm install @mermaid-js/mermaid-cli
echo "Installed as ./node_modules/.bin/mmdc"
echo "Mermaid-cli installed, version" $(./node_modules/.bin/mmdc --version)

# Move the node_modules/ folder out of docs/ to prevent docs/node_modules/**/README.md getting
# included by sphinx. This approach also avoid having change conf.py to explicitly ignore
# node_modules/**/README.md.
mv node_modules/ ..
mkdir -p _build/bin
[[ -L _build/bin/mmdc ]] && rm _build/bin/mmdc
ln -s ../../../node_modules/.bin/mmdc _build/bin/mmdc
