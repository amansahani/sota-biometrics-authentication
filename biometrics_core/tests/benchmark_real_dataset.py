"""
Benchmark Test on Real Open-Source Retinal Fundus Dataset (RetinaMNIST).
Tests:
1. Loading real fundus images from open source dataset
2. Vessel enhancement & normalization
3. Retinal feature extraction with SOTA RetinaVesselExtractor
4. Cancelable BioHashing vault template generation
5. Same-identity vs different-identity matching & separation score
"""

import os
import sys
import numpy as np
import cv2
import torch
from datasets import load_dataset
from PIL import Image

from biometrics_core.preprocessing.processors import RetinaPreprocessor
from biometrics_core.models.extractors import RetinaVesselExtractor
from biometrics_core.security.vault import CancelableBiometricVault
from biometrics_core.fusion.engine import MultimodalFusionEngine
from biometrics_core.liveness.pad import BiometricLivenessDetector


def run_benchmark():
    print("==================================================")
    print("  RUNNING BENCHMARK ON REAL OPEN-SOURCE DATASET   ")
    print("  Dataset: danjacobellis/retinamnist_224 (Fundus) ")
    print("==================================================")

    # 1. Stream 5 real retinal fundus images
    print("\n[Step 1] Streaming real retinal fundus scans...")
    ds = load_dataset("danjacobellis/retinamnist_224", split="train", streaming=True)
    samples = []
    for idx, item in enumerate(ds):
        samples.append(item["image"])
        if len(samples) >= 5:
            break

    print(f"✓ Fetched {len(samples)} real patient fundus images (resolution: {samples[0].size})")

    # 2. Preprocess with CLAHE & Green Spectrum Isolation
    print("\n[Step 2] Processing real fundus images through CLAHE & Vessel extraction...")
    prep = RetinaPreprocessor()
    pad = BiometricLivenessDetector()
    processed_images = []
    
    for i, pil_img in enumerate(samples):
        # Convert PIL to BGR numpy
        bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
        
        # Test real biological PAD / Liveness
        is_live, score, meta = pad.retina_liveness(bgr)
        enhanced = prep.enhance(bgr)
        processed_images.append((bgr, enhanced))
        print(f"  • Image #{i+1}: Biological Hemoglobin Liveness Score = {score:.2f} (Passed: {is_live})")

    # 3. Neural Feature Extraction with SOTA RetinaVesselExtractor
    print("\n[Step 3] Deep feature extraction using SOTA Residual Vessel Extractor...")
    extractor = RetinaVesselExtractor(embedding_dim=512).eval()
    vault = CancelableBiometricVault(512, 512)
    templates = []

    with torch.no_grad():
        for i, (_, enh) in enumerate(processed_images):
            # Input shape: [1, 1, H, W]
            tensor = torch.from_numpy(enh).unsqueeze(0).unsqueeze(0).float() / 255.0
            raw_emb = extractor(tensor).cpu().numpy()[0]
            # Generate privacy-preserving cancelable template
            template = vault.create_cancelable_template(raw_emb, user_id=f"SUBJECT_{i+1:03d}")
            templates.append(template)

    print(f"✓ Extracted 512-D L2-normalized embeddings & generated cancelable templates.")

    # 4. Intra-Class vs Inter-Class Distance Benchmark
    print("\n[Step 4] Matching & Discrimination Benchmark:")
    fusion = MultimodalFusionEngine()

    # Intra-class (Same subject with slight noise / perturbation simulating re-scan)
    noise_sim = processed_images[0][1].copy().astype(np.float32)
    noise_sim += np.random.normal(0, 3, noise_sim.shape)
    noise_sim = np.clip(noise_sim, 0, 255).astype(np.uint8)
    
    with torch.no_grad():
        t_sim = torch.from_numpy(noise_sim).unsqueeze(0).unsqueeze(0).float() / 255.0
        emb_sim = extractor(t_sim).cpu().numpy()[0]
        template_rescan = vault.create_cancelable_template(emb_sim, user_id="SUBJECT_001")

    # Match same subject (Rescan)
    same_match = fusion.verify({"retina": template_rescan}, {"retina": templates[0]})
    print(f"  • Genuine Match (Same Subject Rescan):")
    print(f"    - Confidence Score: {same_match['fused_confidence']:.4f}")
    print(f"    - Authentication: {same_match['aadhaar_compliance_verdict']}")

    # Match different subjects (Inter-class separation)
    diff_scores = []
    for j in range(1, len(templates)):
        diff_match = fusion.verify({"retina": templates[0]}, {"retina": templates[j]})
        diff_scores.append(diff_match['fused_confidence'])

    avg_imposter_score = float(np.mean(diff_scores))
    print(f"\n  • Imposter Matches (Different Subjects):")
    print(f"    - Average Imposter Confidence: {avg_imposter_score:.4f}")
    print(f"    - Max Imposter Score: {max(diff_scores):.4f}")
    print(f"    - Min Imposter Score: {min(diff_scores):.4f}")

    margin = same_match['fused_confidence'] - avg_imposter_score
    print(f"\n✓ Genuine vs Imposter Discriminability Margin: +{margin:.4f}")
    assert same_match['authenticated'], "Genuine match failed threshold"
    assert all(s < 0.70 for s in diff_scores), "False Acceptance detected!"

    print("\n==================================================")
    print(" REAL DATASET BENCHMARK COMPLETED SUCCESSFULLY!   ")
    print("==================================================")


if __name__ == "__main__":
    run_benchmark()
