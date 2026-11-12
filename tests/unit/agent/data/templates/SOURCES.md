# Vendor chat templates kept as test fixtures

Each file is the vendor's `chat_template.jinja`, copied byte for byte at the revision named, so that a dialect's
tests render the template the model was trained with and parse back what it writes. Nothing in them is changed.

| file | source | revision | licence |
|---|---|---|---|
| `GLM-4.7-Flash.jinja` | https://huggingface.co/zai-org/GLM-4.7-Flash | `7dd20894a642a0aa287e9827cb1a1f7f91386b67` | MIT, as declared on the model card (`license: mit`); the repository has no LICENSE file and names no copyright holder beyond its owner, zai-org |
| `Devstral-Small-2-24B-Instruct-2512.jinja` | https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512 | `55c5b41e98c2dbd21b0c8afffc540dcfc9eb5128` | Apache-2.0, as declared on the model card (`license: apache-2.0`), the licence of this repository |

## The MIT licence of `GLM-4.7-Flash.jinja`

Copyright (c) zai-org

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
