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

# No "from __future__ import annotations" to prevent the pyupgrade pre-commit hooks from re-writing
# the type annotations.
from typing import Any, Dict, List, Optional, TypedDict


# NOTES: for typeguard.check_types() to work work, use the lowest common denominator, python-3.8.
# NOTES: FYI, when the time has come to bump the minimum python version, consult the following:
# - python-3.8:
#   * MUST: List, Dict, Optional[str]
# - python-3.9:
#   * CAN: list, dict
#   * MUST: Optional[str]
# - python-3.10:
#   * CAN: list, dict, "None | str"
class MetadataDict(TypedDict):
    states: List[str]
    actions: List[str]
    rewards: List[str]
    forced_categories: Optional[List[str]]
    frequency: Optional[str]
    tags: Dict[str, Any]
