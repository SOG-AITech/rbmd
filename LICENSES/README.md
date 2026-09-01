# Third-Party Licenses

This directory contains license notices for third-party source code and source archives distributed with RBMD2.
The project license is the root-level LICENSE file.

| Component | Version | Distributed location | License files |
|---|---:|---|---|
| nlohmann/json | 3.12.0 | src/common/json.hpp | nlohmann-json-3.12.0/ |
| mio | vendored single header | src/common/mio.hpp | mio-LICENSE.txt |
| cxxopts | 3.1.1 | tools/cxxopts-3.1.1.tar.gz | cxxopts-3.1.1-LICENSE.txt |
| FTXUI | 6.1.9 | tools/ftxui-6.1.9.tar.gz | ftxui-6.1.9-LICENSE.txt |
| spdlog | 1.14.1 | tools/spdlog-1.14.1.tar.gz | spdlog-1.14.1-LICENSE.txt |
| fmt bundled by spdlog | bundled with spdlog 1.14.1 | tools/spdlog-1.14.1.tar.gz | fmt-bundled-with-spdlog-1.14.1-LICENSE.rst |

nlohmann/json 3.12.0 includes third-party portions covered by the additional license texts and provenance information in its subdirectory.
System-provided build and runtime dependencies are not copied into this directory because RBMD2 does not redistribute them here.
