# Fursee License

*Last Updated: July 22, 2026*

Thank you for using Fursee. This Fursee Software License Agreement (hereinafter referred to as the "Agreement") constitutes a legal agreement between you (as an individual or a single entity) and the author and copyright owner of Fursee (hereinafter referred to as the "Author") regarding the downloading, installation, use, copying, and distribution of the Fursee project (including its code, model weights, documentation, and related materials, hereinafter referred to as the "Software").

## 1. Grant of Rights

Subject to the terms and conditions of this Agreement, the Author hereby grants you a royalty-free, non-exclusive, non-transferable license to:

- Freely use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software.
- Use the Software for any purpose, including commercial research and development, and integration into paid products or services.
- **Condition**: The exercise of all the aforementioned rights is strictly contingent upon including a clear attribution to the original author and source in all copies or substantial portions of the Software.

## 2. Privacy Protection & Data Usage Guidelines

Given that the Software involves sensitive functions such as image recognition, users must strictly adhere to the following guidelines:

- **Default Authorization**: By using this model, you implicitly authorize the model to read and process your photographs, strictly limited to image recognition, classification, and related functions.
- **Local Execution & Privacy Commitment**: The Author commits that the Software's own code logic does not contain any mechanisms for uploading data to the cloud. **However, please note that the Author makes no express or implied warranties regarding any data behaviors or privacy risks arising from third-party dependency libraries (e.g., torch, transformers, etc.).**
- **Developer Code of Conduct**: During actual deployment, all photographs processed must be used strictly within the local system scope. Random dissemination, forwarding, or public disclosure is strictly prohibited.
- **Prohibition of Abuse**: You shall not use this model to engage in any activities that infringe upon the privacy of others or violate applicable laws and regulations.

## 3. Patent Grant & Defense

- **Patent Grant**: The Author hereby grants you a global, royalty-free, non-exclusive patent license to use, copy, modify, and distribute the Software.
- **Patent Retaliation Clause**: If you institute a patent infringement lawsuit against any party (including the Author) claiming that the Software constitutes direct or indirect patent infringement, all patent grants and rights granted to you under this Agreement shall automatically and immediately terminate.

## 4. Disclaimer & Limitation of Liability

- **"As Is" Warranty Exclusion**: The Software is provided on an "AS IS" basis, without warranties of any kind, express or implied, including merchantability, fitness for a specific purpose, non-infringement and error-free operation. This disclaimer shall not apply to losses caused by the Author’s intentional misconduct or gross negligence, which cannot be excluded by mandatory law.
- **Model Result Disclaimer**: All recognition outputs are for reference only. The Author does not guarantee accuracy, integrity or correctness of analysis results. All direct/indirect losses arising from recognition misjudgment shall be borne by you.
- **Liability Cap**: To the maximum extent permitted by applicable law, the Author’s total aggregate liability under this Agreement shall not exceed the total amount you actually paid to the Author (zero for free open-source use). The Author shall not be liable for consequential, indirect, special damages including lost profit, business interruption, mass data loss.
- **User Pre-Deployment Responsibility**: You shall complete full testing, security audit and legal compliance evaluation before commercial deployment of the Software. The Author does not warrant that the Software complies with all regional data privacy, copyright and cyber laws globally; you shall confirm local legal eligibility independently.
- **Indemnification**: If your non-compliant use leads to third-party lawsuits, administrative penalties or compensation claims, you shall fully indemnify, defend and hold harmless the Author from all damages, legal fees, fines and settlement costs.

## 5. Supplemental Terms for Derivative Commercial Works

### Intellectual Property Ownership of Derivatives

This Software is built upon third-party open-source frameworks YOLO and DINOv3. All intellectual property rights belonging to the original YOLO, DINOv3 source code, pre-trained base weights shall remain with their respective original copyright holders and are governed by their respective open-source licenses.

The Author solely holds copyright for self-written project engineering scripts, pipeline logic, customized data processing code, and fine-tuned adjusted model weights derived from the above base models.

For any independent, self-developed business logic, module and functional code newly created by you and separated from the core algorithm files of the Software (hereinafter referred to as Your Independent Code), you shall hold complete intellectual property rights of Your Independent Code. When exercising rights over derivative works containing both the Software and Your Independent Code, you must fully comply with all attribution and usage obligations under this Agreement, and shall not remove or obscure attribution notices for the original upstream frameworks, the Author’s adjusted model weights, and this project.

### Distinction Between Open-Source and Closed-Source Distribution

You are permitted to modify the Software and distribute modified copies in closed-source binary form for paid commercial products, SaaS services and offline sales, subject to the following mandatory additional obligations for closed-source commercial distribution:

- Retain original copyright headers, license statements for upstream YOLO/DINOv3 and this project in all unmodified source files of the Software;

- Display clear, readable attribution information covering YOLO, DINOv3 and this project on product About page, startup console log or official documentation of all closed-source commercial products integrated with the Software;

- Transmit the full text of this Agreement and all attached obligations to all downstream sublicensees, distributors and end commercial customers, and remind downstream parties to comply with the original open-source terms of YOLO and DINOv3.

If you distribute derivative works in open-source source code form, you may adopt any license compatible with this Agreement as well as the original licenses of YOLO and DINOv3, but you must reserve all multi-party attribution requirements and attach a copy of this Agreement together with the open-source distribution package.

### No Copyleft / Source Code Contagion Rule

This Agreement does not contain full-work copyleft (code contagion) provisions:

- Only this project’s self-written code, fine-tuned weights, documents and files directly modified from this Software are bound by this Agreement; the original YOLO/DINOv3 base materials continue to follow their respective original licenses;

- Your Independent Code developed separately and interacted with the Software via standardized APIs, interfaces or separate module calls shall not be subject to the terms of this Agreement, and you may keep such code fully closed-source without mandatory public disclosure;

- Static linking, dynamic linking, packaging and embedding the Software into your commercial product shall not force you to open-source your entire commercial product codebase. You remain solely responsible for verifying compliance with YOLO and DINOv3’s license terms when conducting closed-source commercialization.

## 6. Trademark Provision

Nothing in this License grants you the right to use "Fursee" name, logo, project brand for commercial marketing, product trademark or official endorsement without prior written consent from the Author. You may only use the name solely for required attribution statement.

## 7. Termination

If you fail to comply with any terms of this Agreement (including, but not limited to, failing to provide attribution, violating privacy guidelines, or triggering the patent retaliation clause), all rights granted to you under this Agreement shall **automatically and immediately terminate**, without any cure period or prior notice.

## 8. Modification of This License

- The Author may update this License by committing new LICENSE.md to the GitHub repository root, with updated last-updated date.

- Updated terms only apply to new downloads/clones after the release date; all existing copies and derivatives downloaded before update continue to follow the original License version at download time.

- The Author shall post a clear change log in README.md when updating license clauses.