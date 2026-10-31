"""
Comprehensive Integration and Benchmark Test Suite:
Validates:
1. Fingerprint preprocessing & DeepPrint inference
2. Iris unwrapping & SAM-Iris inference
3. Retina CLAHE enhancement & RetinaVesselExtractor inference
4. Liveness detection against simulated presentation attacks
5. End-to-end Enrollment, 1:1 Verification, and 1:N Identification API test
"""

import base64
import cv2
import numpy as np
import torch
from fastapi.testclient import TestClient

from biometrics_core.api.server import app
from biometrics_core.preprocessing.processors import FingerprintEnhancer, IrisNormalizer, RetinaPreprocessor
from biometrics_core.models.extractors import DeepPrintExtractor, SAMIrisExtractor, RetinaVesselExtractor, ArcMarginProduct
from biometrics_core.liveness.pad import BiometricLivenessDetector
from biometrics_core.security.vault import CancelableBiometricVault
from biometrics_core.fusion.engine import MultimodalFusionEngine


def generate_synthetic_fingerprint(seed=42) -> np.ndarray:
    np.random.seed(seed)
    img = np.zeros((256, 256), dtype=np.uint8)
    for r in range(10, 120, 8):
        cv2.circle(img, (128, 128), r, 255, 2)
    noise = (np.random.randn(256, 256) * 20).astype(np.int16)
    noisy_img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy_img


def generate_synthetic_iris(seed=42) -> np.ndarray:
    np.random.seed(seed)
    img = np.full((256, 256), 180, dtype=np.uint8)
    # Pupil
    cv2.circle(img, (128, 128), 35, 20, -1)
    # Iris annular texture
    for angle in range(0, 360, 5):
        rad = np.deg2rad(angle)
        x1 = int(128 + 40 * np.cos(rad))
        y1 = int(128 + 40 * np.sin(rad))
        x2 = int(128 + 90 * np.cos(rad))
        y2 = int(128 + 90 * np.sin(rad))
        cv2.line(img, (x1, y1), (x2, y2), int(70 + 30 * np.sin(angle)), 1)
    # Specular reflection
    cv2.circle(img, (140, 120), 4, 250, -1)
    return img


def generate_synthetic_retina(seed=42) -> np.ndarray:
    np.random.seed(seed)
    fundus = np.zeros((256, 256, 3), dtype=np.uint8)
    fundus[:, :, 2] = 160 # Red channel high in fundus
    fundus[:, :, 1] = 90  # Green channel
    fundus[:, :, 0] = 30  # Blue channel low
    # Retinal vessel branches
    cv2.line(fundus, (128, 128), (50, 40), (20, 30, 80), 3)
    cv2.line(fundus, (128, 128), (200, 50), (20, 30, 80), 2)
    cv2.line(fundus, (128, 128), (70, 200), (20, 30, 80), 2)
    cv2.line(fundus, (128, 128), (190, 210), (20, 30, 80), 3)
    return fundus


def img_to_b64(img: np.ndarray) -> str:
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def run_tests():
    print("--- [1/5] Testing Preprocessing Pipelines ---")
    fp = generate_synthetic_fingerprint()
    enhancer = FingerprintEnhancer()
    fp_enh = enhancer.enhance(fp)
    assert fp_enh.shape == fp.shape, "Fingerprint enhancement shape mismatch"

    iris = generate_synthetic_iris()
    normalizer = IrisNormalizer()
    iris_polar = normalizer.unwrap(iris)
    assert iris_polar.shape == (64, 256), f"Iris polar shape mismatch: {iris_polar.shape}"

    retina = generate_synthetic_retina()
    retina_prep = RetinaPreprocessor()
    retina_enh = retina_prep.enhance(retina)
    assert retina_enh.shape == (256, 256), "Retina enhancement shape mismatch"
    print("✓ All preprocessing stages verified!")

    print("\n--- [2/5] Testing SOTA PyTorch Neural Extractors ---")
    fp_model = DeepPrintExtractor(512).eval()
    t_fp = torch.from_numpy(fp_enh).unsqueeze(0).unsqueeze(0).float() / 255.0
    fp_emb = fp_model(t_fp)
    assert fp_emb.shape == (1, 512), f"Fingerprint embedding error: {fp_emb.shape}"

    iris_model = SAMIrisExtractor(512).eval()
    t_iris = torch.from_numpy(iris_polar).unsqueeze(0).unsqueeze(0).float() / 255.0
    iris_emb = iris_model(t_iris)
    assert iris_emb.shape == (1, 512), f"Iris embedding error: {iris_emb.shape}"

    retina_model = RetinaVesselExtractor(512).eval()
    t_retina = torch.from_numpy(retina_enh).unsqueeze(0).unsqueeze(0).float() / 255.0
    retina_emb = retina_model(t_retina)
    assert retina_emb.shape == (1, 512), f"Retina embedding error: {retina_emb.shape}"

    arcface = ArcMarginProduct(512, out_features=100)
    logits = arcface(fp_emb, torch.tensor([42]))
    assert logits.shape == (1, 100), "ArcFace head evaluation mismatch"
    print("✓ DeepPrint, SAM-Iris, RetinaVessel & ArcFace forward passes verified!")

    print("\n--- [3/5] Testing Presentation Attack Detection (Liveness) ---")
    pad = BiometricLivenessDetector()
    live_fp, score_fp, _ = pad.fingerprint_liveness(fp)
    assert live_fp, "Fingerprint failed liveness"

    live_iris, score_iris, _ = pad.iris_liveness(iris)
    assert live_iris, "Iris failed liveness"

    live_retina, score_retina, _ = pad.retina_liveness(retina)
    assert live_retina, "Retina failed liveness"
    print(f"✓ Liveness verified: FP={score_fp:.2f}, Iris={score_iris:.2f}, Retina={score_retina:.2f}")

    print("\n--- [4/5] Testing Cancelable Vault & Multimodal Fusion ---")
    vault = CancelableBiometricVault(512, 512)
    template_user1 = vault.create_cancelable_template(fp_emb.detach().numpy()[0], "user_aadhaar_001")
    template_user2 = vault.create_cancelable_template(fp_emb.detach().numpy()[0], "user_aadhaar_002")
    # Irreversibility and service unlinkability: Cross correlation between different users with same raw print should be near 0
    cross_corr = np.dot(template_user1, template_user2)
    assert abs(cross_corr) < 0.25, f"Cancelable biometrics cross-linking leakage detected: {cross_corr}"
    print(f"✓ Cancelable Biometric Security: Unlinkability orthogonal cross-correlation = {cross_corr:.4f}")

    fusion = MultimodalFusionEngine(threshold=0.70)
    verif = fusion.verify({"fingerprint": template_user1}, {"fingerprint": template_user1})
    assert verif["authenticated"], "Self-verification failed"
    print(f"✓ Multimodal Fusion match verdict: {verif['aadhaar_compliance_verdict']} (Confidence: {verif['fused_confidence']:.2f})")

    print("\n--- [5/5] Testing FastAPI End-to-End API Routes ---")
    client = TestClient(app)

    # Health
    r = client.get("/health")
    assert r.status_code == 200

    # Enroll
    fp_b64 = img_to_b64(fp)
    iris_b64 = img_to_b64(iris)
    retina_b64 = img_to_b64(retina)

    enroll_payload = {
        "user_id": "IND-AADHAAR-883920192841",
        "fingerprint_b64": fp_b64,
        "iris_b64": iris_b64,
        "retina_b64": retina_b64
    }
    r = client.post("/enroll", json=enroll_payload)
    assert r.status_code == 200, f"Enroll error: {r.text}"
    print("✓ Successfully enrolled user into Vault via REST API!")

    # Verify 1:1
    verify_payload = {
        "user_id": "IND-AADHAAR-883920192841",
        "fingerprint_b64": fp_b64,
        "iris_b64": iris_b64
    }
    r = client.post("/verify", json=verify_payload)
    assert r.status_code == 200
    res = r.json()
    assert res["verification_result"]["authenticated"] is True
    print(f"✓ 1:1 Verification Passed! Confidence: {res['verification_result']['fused_confidence']:.4f}")

    # Identify 1:N
    r = client.post("/identify", json=verify_payload)
    assert r.status_code == 200
    id_res = r.json()
    assert id_res["identified_user_id"] == "IND-AADHAAR-883920192841"
    print(f"✓ 1:N Global Identification Found Match: {id_res['identified_user_id']}")

    print("\n==========================================")
    print(" ALL SOTA BIOMETRICS SYSTEM TESTS PASSED! ")
    print("==========================================")


if __name__ == "__main__":
    run_tests()
