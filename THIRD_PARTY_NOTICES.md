CLAPSep — Third-Party Notices
Copyright (c) 2024 Hao Ma

This product is licensed under the MIT License (see LICENSE).

This product includes software developed by third parties. The notices below
are provided in accordance with the MIT License requirement that the copyright
notice and permission notice of the upstream works be included in all copies or
substantial portions of the Software.

================================================================================
1. LAION-CLAP  (https://github.com/LAION-AI/CLAP)
================================================================================

Used as:  external PyPI dependency  (pip: laion-clap), declared in
          requirements.txt.

Included here because:
  * model/CLAPSep.py imports the `laion_clap` package and calls its public
    API (CLAP_Module / get_audio_embedding_from_data / get_text_embedding);
  * model/CLAPSep_decoder.py performs `from laion_clap.clap_module.htsat
    import *` and reuses and adapts the upstream HTS-AT/Swin decoder building
    blocks (class signatures, constructor parameter sets and the
    `HTSAT_Decoder` docstring derive from `clap_module/htsat.py`).

Upstream license:
  * The LAION-CLAP package (PyPI `laion-clap`) is distributed under
    CC0 1.0 Universal (Public Domain Dedication). The wheel ships this text
    as `laion_clap-<version>.dist-info/licenses/LICENSE`.
  * NOTE: the PyPI metadata of `laion-clap` also carries the classifier
    "License :: OSI Approved :: Apache Software License". That classifier is
    inconsistent with the CC0 1.0 license text actually shipped in the
    package. The CC0 1.0 text is the authoritative license.
  * CC0 1.0 imposes no conditions on reuse, so it is compatible with, and
    imposes no obligations upon, the MIT licensing of this product.

The LAION-CLAP codebase itself builds on third-party components that are
carried inside the `laion_clap` package:

1a. HTS-AT (Hierarchical Token-Semantic Audio Transformer)
    Author: Ke Chen <knutchen@ucsd.edu>
    Source: https://github.com/laion-ai/CLAP (clap_module/htsat.py)
    The file header of clap_module/htsat.py states that its layers are
    "based and referred from https://github.com/microsoft/Swin-Transformer".

1b. Microsoft Swin-Transformer
    Copyright (c) 2021 Microsoft
    Source: https://github.com/microsoft/Swin-Transformer
    License: MIT License

1c. open_clip
    Copyright (c) 2012-2021 Scott Lundberg, Leo Dirac, Romain Beaumont
    and contributors
    Source: https://github.com/mlfoundations/open_clip
    License: MIT License
    (The LAION-CLAP README states that it adopts the open_clip codebase.)

Because this product links against and adapts code that is derived from the
MIT-licensed HTS-AT / Swin-Transformer / open_clip works, the following
notice is reproduced:

    MIT License

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the
    "Software"), to deal in the Software without restriction, including
    without limitation the rights to use, copy, modify, merge, publish,
    distribute, sublicense, and/or sell copies of the Software, and to permit
    persons to whom the Software is furnished to do so, subject to the
    following conditions:

    The above copyright notice and this permission notice shall be included
    in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
    OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
    IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
    CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
    TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
    SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

================================================================================
2. TorchLibrosa  (https://github.com/qiuqiangkong/torchlibrosa)
================================================================================
Copyright (c) Qiuqiang Kong
Used as: external PyPI dependency (pip: torchlibrosa), declared in
         requirements.txt. Used via `from torchlibrosa import ISTFT, STFT,
         SpecAugmentation` and `from torchlibrosa.stft import magphase`.
License: MIT License

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the
    "Software"), to deal in the Software without restriction, including
    without limitation the rights to use, copy, modify, merge, publish,
    distribute, sublicense, and/or sell copies of the Software, and to permit
    persons to whom the Software is furnished to do so, subject to the
    following conditions:

    The above copyright notice and this permission notice shall be included
    in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
    OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
    IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
    CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
    TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
    SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

================================================================================
3. AudioSep  (https://github.com/Audio-AGI/AudioSep)
================================================================================
Copyright (c) Xubo Liu
The audio reconstruction routine in model/CLAPSep.py
(`LightningModule.wav_reconstruct`) is adapted from
`models/resunet.py` in the AudioSep project, as noted by the in-source
reference comment.
License: MIT License

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the
    "Software"), to deal in the Software without restriction, including
    without limitation the rights to use, copy, modify, merge, publish,
    distribute, sublicense, and/or sell copies of the Software, and to permit
    persons to whom the Software is furnished to do so, subject to the
    following conditions:

    The above copyright notice and this permission notice shall be included
    in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
    OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
    IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
    CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
    TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
    SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

================================================================================
4. Other dependencies
================================================================================
Other runtime dependencies (see requirements.txt), including PyTorch,
torchaudio, torchvision, librosa, numpy, einops, loralib, transformers and
pytorch-lightning, are used as unmodified external libraries and are NOT
vendored into this repository. Each remains under its own upstream license;
those licenses are not reproduced here. Consult the individual packages for
their terms (for example, PyTorch is BSD-3-Clause, librosa is ISC, and
einops/loralib are MIT).
