# YOLO for Medieval OMR: Project Brief
## Project Goal
Experiment with YOLO’s object detection capabilities to identify and localize musical elements (text, staves, neumes/symbols) on medieval manuscript pages, with the ultimate goal of producing OMR’d pages with accurate bounding boxes. We want to evaluate whether YOLO can outperform our current document analysis pipeline.

Target output: Detected symbols with bounding boxes indicating position on page (text-music-staves separation and symbol localization).
 
## Recommended Approach
### Primary Architecture: DocLayout-YOLO
Start with DocLayout-YOLO rather than basic YOLO tutorials. It’s purpose-built for document layout analysis with key features directly applicable to medieval manuscripts:

- Global-to-Local Controllable Receptive Module (GL-CRM): Handles multi-scale variation (tiny neume components → large, decorated initials)
Synthetic data generation pipeline: Adaptable for creating training data with parchment damage/degradation

- Document-specific optimizations: Built on YOLOv10/11 with modifications for complex document structures

### Key Considerations for Medieval Manuscripts
1.	Multi-scale detection: Significant scale variation between small neumes, wide staves, and large initials
    - 	GL-CRM architecture addresses this
    -   Consider anchor-free approaches for varied neume shapes
2.	Class imbalance: Some neume types are much rarer, or resemble text items; some text abbreviations resemble musical glyphs
    -   Plan for weighted loss functions
    -   Data augmentation strategies
3.	Spatial relationships: Medieval notation has strict rules (neumes on staves, text below)
    -   Pure YOLO won’t capture these relationships
    -   Consider hybrid approach: YOLO detection + post-processing with paleographic rules?
4.	Parchment degradation: Fading, bleed-through, damage
    -   DocLayout-YOLO’s synthetic data approach can generate realistic degradation patterns
    -   Important for robust training
5.	Dataset strategy:
    -   Manuscript-aware cross-validation (don’t split within manuscripts)
    -   Document-specific fine-tuning likely needed
 
## GitHub Repositories
### Primary (Start Here):
DocLayout-YOLO: https://github.com/opendatalab/DocLayout-YOLO 
- Production-ready, 2k+ stars, actively maintained
- Built on YOLOv10/11 with document-specific optimizations
### External Reference Datasets (Optional):
OMMR4all Medieval Datasets: https://github.com/OMMR4all/datasets 
- Graduel de Nevers (12th c., 49 pages)
- Pa904 (12th-13th c., 20 pages)
- Classes: clefs, accidentals, neume components (starts, gapped, looped)
- Note: Useful for comparison/benchmarking, but we have our own annotated dataset

**Existing DDMAL annotations**
- Gen and Kyrie have a ton of images
- Kyrie has tens of thousands of neume crops, you can find a lot of square notation ones here: https://huggingface.co/datasets/grackle-in-a-HEB-parking-lot/possumm-local

### Reference:
- Base framework (DocLayout-YOLO builds on this)
 
## Key Papers
### Core Architecture & Methodology
1.	“DocLayout-YOLO: Enhancing Document Layout Analysis through Diverse Synthetic Data and Global-to-Local Adaptive Perception”
	- Zhao et al., arXiv:2410.12628, October 2024
	- https://arxiv.org/abs/2410.12628
	- Read this first - foundation architecture
2.	“Optical Medieval Music Recognition Using Background Knowledge”
	- Hartelt & Puppe, MDPI Algorithms 2022
	- https://www.mdpi.com/1999-4893/15/7/221
 	- Domain-specific: medieval notation, post-processing with musicological rules
  	- Uses YOLO for symbol detection, includes dataset info
3.	“Understanding Optical Music Recognition”
	- Calvo-Zaragoza et al., 2020
	- General OMR overview (good context)

**Historical Document Detection (Techniques Transfer)**

4.	“YOLO-HTR: Page-Level Recognition of Historical Handwritten Document Collections”
	- Lomov et al., 2024
	- Springer AIST 2024 proceedings
	- Combines YOLO with text recognition for historical documents
5.	“A Historical Handwritten French Manuscripts Text Detection Method in Full Pages”
	- MDPI Information, August 2024
	- https://www.mdpi.com/2078-2489/15/8/483
	- YOLOv8s modifications for complex/degraded text

**YOLO Evolution (Architecture Context)**

6.	“Ultralytics YOLO Evolution: An Overview of YOLO26, YOLO11, YOLOv8, and YOLOv5”
	- arXiv:2510.09653, 2025
	- Recent architectural innovations: 
	- YOLO11: C3k2 bottlenecks, C2PSA attention (small object performance)
	- YOLO26: NMS-free inference, Progressive Loss Balancing, Small-Target-Aware Label Assignment
7.	“A Decade of You Only Look Once (YOLO) for Object Detection”
	- arXiv review paper, 2025
	- https://arxiv.org/abs/2504.18586
	- Comprehensive YOLO evolution overview

**OMR-Specific YOLO Applications**

8.	“State-of-the-Art Model for Music Object Recognition with Deep Learning”
	- MDPI Applied Sciences, 2019
	- https://www.mdpi.com/2076-3417/9/13/2645
	- YOLO/Darknet53 for note recognition, upsampling strategies
9.	Pacha et al. - “Optical Music Recognition in Mensural Notation with Region-Based Convolutional Neural Networks”
	- https://www.researchgate.net/publication/327962576_Optical_Music_Recognition_in_Mensural_Notation_with_Region-Based_Convolutional_Neural_Networks
	- Used Faster R-CNN and YOLO for mensural notation (16th-18th c.)
	- Treated OMR as object detection task
 
### Additional Useful Information
**Key Advantage: We (Will) Have An Annotated Dataset**

Having our own annotated text-music-staves dataset is a major advantage. This means:
- Controlled evaluation: Can directly compare YOLO performance against Rodan’s current pipeline on the same data
- Flexibility: Can add more granular annotations incrementally if needed

Critical first steps:
1.	Verify annotation format (YOLO needs: class_id x_center y_center width height per line, normalized 0-1)
	DONE
2.	Document your class definitions clearly
    - Text (all text present on a page, including initials, excepting illuminated or exceptionally decorated large initials. This includes rubrics!)
    - Music (all music symbols present on a page, including clef and custos)
    - Staves (all four lines making a stave collected in the same bounding box. YOLO does not have an issue with overlapping bounding boxes. In future this may be tweaked)
3.	Analyze class distribution for severe imbalance
    Dataset creation is currently striving for 500 glyphs/manuscript
4.	Ensure manuscript-aware splits (entire manuscripts in train OR val OR test, not split across)
    This will be very important for validation. 
    If anyone needs help with this, please talk to Kyrie, or see this method from POSSUMM. Alternatively, use a chappy.

Sample from produced annotation.txt file, from Mothra annotator:

```
0 0.139432 0.215570 0.094260 0.024340
0 0.244700 0.216487 0.108447 0.022994
0 0.338796 0.212512 0.078767 0.023606
0 0.446755 0.215020 0.120515 0.022016
0 0.557404 0.215386 0.099152 0.024706
0 0.629159 0.215020 0.041748 0.026663
```

For a sample of a fully annotated medieval music page, YOLO.txt output, and UI view of a completely annotated page, see Mothra/samples/ Aarau_MsMurF2_5v.

## Medieval-Specific Challenges
### Neume notation characteristics:
	- Neumes represent melodic gestures (1+ notes per symbol)
	- Square notation (12th-13th c. onward) vs. earlier staffless forms
	- High variability in scribal hands, house styles, and regions
## Dataset Preparation
### Mothra annotated dataset:
1) Verify annotation format compatibility with YOLO (YOLO expects: class_id, x_center, y_center, width, height - all normalized)
    Done!
2) Check class distribution for imbalance issues
    ONGOING
3) Decide on train/validation/test splits (manuscript-aware - don’t split pages from same manuscript)
4) Consider if dataset size is sufficient or if synthetic augmentation is needed
5) Converting existing annotations to YOLO format:
    DocLayout-YOLO expects standard YOLO format (txt files with one line per object)
    If your annotations are in a different format (e.g., COCO JSON, Pascal VOC XML, custom format), you’ll need conversion scripts
        Mothra/annotator/json2yolo.py
    Ultralytics has built-in converters for common formats
### Dataset characteristics to document:
	- Total pages/images
	- Number of classes (text, music, staves, specific symbol types?)
	- Annotation granularity (region-level vs. symbol-level)
	- Manuscript types/periods covered
	- Image resolution and quality variations
	- Degradation patterns present

## Technical Recommendations
**Architecture modifications to explore:**
- Attention mechanisms (YOLO11’s C2PSA) for fine-grained distinctions between similar neume types
- Small-target optimization (YOLO26’s STAL)
- Anchor-free detection (better for irregular neume shapes?)
**Training strategies:**
- Adapt DocLayout-YOLO’s Mesh-candidate BestFit for synthetic data generation
- Augment with parchment-specific degradation (fading, bleed-through, tears)
- Implement manuscript-aware train/val splits
- Consider pre-training on DocSynth-300K before fine-tuning on medieval data
**Post-processing**:
- YOLO outputs bounding boxes + class predictions
- Add rule-based refinement using musicological constraints: 
	- Neumes must align with staff lines
	- Spatial relationships between clefs, neumes, text
	- Melodic constraints (if incorporating domain knowledge)

### Evaluation Metrics
**Standard object detection metrics:**
- mAP (mean Average Precision)
- Precision/Recall per class
- IoU (Intersection over Union)
**OMR-specific (if extending to full transcription):**
- Symbol Error Rate
- Melody Accuracy Rate (mAR) - edit distance for melodic content

## Development Path
1.	Familiarization: Review DocLayout-YOLO architecture, run demos
2.	Dataset preparation: Examine OMMR4all data, understand annotation format
3.	Baseline training: Train DocLayout-YOLO on medieval dataset
4.	Iteration: Adjust architecture/training based on results
5.	Comparison: Benchmark against current pipeline
6.	Post-processing: Develop rules for spatial relationship enforcement
 
## Quick Start
1.	Clone DocLayout-YOLO: git clone https://github.com/opendatalab/DocLayout-YOLO
2.	Review README and paper (arXiv:2410.12628)
3.	Prepare your dataset: 
        - Convert annotations to YOLO format if needed
        - Organize into train/val/test splits (manuscript-aware)
        - Create dataset YAML config file
4.	Run baseline training on your annotated dataset
5.	Evaluate performance and compare against current pipeline
6.	Iterate on architecture/hyperparameters as needed
 
## Questions to Consider
1) Dataset format: Are our annotations already in YOLO format, or do we need conversion scripts?
	- YES, they are in YOLO format, but the `Mothra` repo also has a JSON -> YOLO script if we have existing data that needs conversion.
2) Class definitions: Do our text-music-staves annotations map cleanly to what we want YOLO to detect? Do we need more granular symbol-level classes?
	- For our current purposes, TMS is working. Subject to change.
3) Dataset size: Is our current annotated dataset sufficient, or should we augment with synthetic data?
	- Our current goal is 500 musical glyphs/manuscript and corresponding text-stave ratio (whole page).
4) Pre-training: Should we pre-train on DocSynth-300K (general documents) before fine-tuning on our medieval manuscripts?
	- Let’s go as “out of the box” as we can initially to capture the real dis/advantages of either approach.
5) Evaluation strategy: How do we evaluate “success” compared to current pipeline? (Speed? Accuracy? Both? Specific error types?)
	- Speed is a not insignificant factor—the full Rodan pipeline particularly including the active training episode with the Interactive Classifier followed by Neon correction takes up an enormous amount of time. If YOLO can accelerate/bypass through separation and the IC that’s already an incredible progression. Accuracy, however, would need to make it worth it: **YOLO needs to be faster and more accurate than we could just manually encode in Neon.** In other words: is YOLO faster than Gen is? 
6) Annotation completeness: Do we need to expand annotations to include more detailed symbol classes for full OMR?
	- This would include labeling for neume types. Can we get away with `music`? Or possibly `music`, `clef`, `custode`, `divisio`? 
