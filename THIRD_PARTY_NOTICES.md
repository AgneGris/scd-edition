# Third-party notices

## Motor Unit Toolbox

SCD Edition contains selected source files from
[Motor Unit Toolbox](https://github.com/imendezguerra/motor_unit_toolbox),
version 1.0, commit `d84dc6c943daa5d686b2911b48b13bf718628130`.
The vendored routines are used to calculate motor-unit properties and spike-train
agreement. Local changes are limited to package-relative imports and extracting
the five MUAP channel-selection helpers used by the property module.

Copyright (c) 2024 Irene Mendez Guerra

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Example recording

The bundled `emg.mat` is copied from Swarm-Contrastive Decomposition commit
`8cd6153788b730383d6d6a1833ee99899ee5bcfb` and is redistributed under that
project's BSD-3-Clause license. Its provenance and checksum are recorded in
`src/scd_app/resources/examples/README.md`.
