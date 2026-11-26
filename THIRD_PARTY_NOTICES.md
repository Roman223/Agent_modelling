# Numerical-method adaptations

The MF-DFA implementation in `src/multifractal.py` adapts the forward/backward
NumPy segmentation in [MFDFA 0.4.3](https://github.com/LRydin/MFDFA), with a
different polynomial projection, a logarithmic q=0 limit, and explicit handling
of zero and numerically unresolved segments. Method reference: Kantelhardt et al.,
Physica A 316 (2002), 87–114, https://doi.org/10.1016/S0378-4371(02)01383-3.

## MFDFA — MIT License

Copyright (c) 2019-2022 Leonardo Rydin Gorjão

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

## pyunicorn — BSD 3-Clause License

The IAAFT spectral/rank projections in `src/multifractal_controls.py` adapt
[pyunicorn 0.8.0](https://www.pik-potsdam.de/~donges/pyunicorn/_modules/pyunicorn/timeseries/surrogates.html).
This adaptation starts from a random permutation, uses local generators and
stable tie ordering, handles zero Fourier coefficients, and records convergence
and amplitude-spectrum mismatch. Method reference: Schreiber and Schmitz,
Physica D 142 (2000), 346–382, https://doi.org/10.1016/S0167-2789(00)00043-9.
Software reference: Donges et al., Chaos 25 (2015), 113101,
https://doi.org/10.1063/1.4934554.

Copyright (C) 2008–2024 Jonathan F. Donges and pyunicorn authors

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
