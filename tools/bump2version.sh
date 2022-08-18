#!/usr/bin/env bash

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

# DO NOT EXECUTE.
# This script documents CLI commands meant to run manually.

# NOTE: add --allow-dirty when testing on dirty repo

# 1. Release: x.y.z-dev => x.y.z
bump2version release --commit --tag-name 'v{new_version}'

# 2. Increase: x.y.z-dev => x'.y'.z'-dev
bump2version major|minor|patch
