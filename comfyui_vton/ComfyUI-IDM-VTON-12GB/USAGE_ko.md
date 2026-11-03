# IDM-VTON · RTX 5070 12GB 설정

## 설치 구성

- ComfyUI: `C:\Dev\StabilityMatrix\Data\Packages\ComfyUI`
- 커스텀 노드: 위 폴더의 `custom_nodes\ComfyUI-IDM-VTON-12GB`
- 워크플로: `user\default\workflows\IDM-VTON-12GB.json`
- 768×1024 워크플로: `user\default\workflows\IDM-VTON-768x1024.json`
- 예제 입력: `input\idm_sample_human.jpg`, `input\idm_sample_garment.jpg`
- 별도 실행 환경과 모델: 이 작업 폴더의 `work\runtime`, `work\models`
- 실제 실행 스크립트: `outputs\IDM-VTON-12GB\runner.py`

**이 작업 폴더를 삭제하거나 이동하지 마세요.** 설치된 연결 노드가 이 안의 실행 환경과 모델을 참조합니다. 기존 ComfyUI의 Python 패키지는 변경하지 않았으며, 별도 환경이 기존 CUDA PyTorch를 읽어 사용합니다. 향후 Stability Matrix에서 ComfyUI의 PyTorch를 교체하면 재검증이 필요할 수 있습니다.

## 실행

1. ComfyUI 재시작과 새 노드 활성화를 완료했습니다. 현재 실행 중인 `http://127.0.0.1:8188`에서 사용할 수 있습니다. 이후에는 평소처럼 Stability Matrix에서 ComfyUI를 실행하면 됩니다.
2. 먼저 `IDM-VTON-전처리확인.json`으로 마스크와 포즈를 확인하고, `IDM-VTON-768x1024.json` 워크플로를 엽니다. 빠른 저해상도 시험은 `IDM-VTON-12GB.json`을 사용합니다. 파일을 캔버스에 드래그해도 됩니다.
3. Person에는 인물 사진, Garment에는 의상 상품 사진을 넣습니다.
4. 전처리 노드의 `clothing_prompt`에는 **인물이 현재 입고 있는 옷**을 적습니다. 예: `shirt`, `jacket`, `dress`.
5. 추론 노드의 `garment_description`에는 **새로 입힐 옷**을 영어로 설명합니다.
6. 768×1024, 20 steps, 배치 1로 실행합니다. 빠른 시험은 384×512로 낮춥니다.

## 마스크 확인

마스크의 흰색 영역을 다시 생성합니다. 얼굴, 머리카락, 손 등 유지할 영역은 검은색이어야 합니다. 자동 SAM 마스크는 기존 옷을 검출하며, 새 옷의 실루엣을 예측하지 않습니다.

- 기존 민소매 → 긴소매: 새 소매가 들어갈 팔 주변을 포함하도록 마스크를 직접 넓혀야 합니다.
- 작은 상의 → 큰 재킷: 몸 바깥쪽 여유 공간도 필요할 수 있습니다.
- `mask_grow`는 전체 마스크를 지정 픽셀만큼 넓힙니다. 얼굴/손까지 침범하지 않도록 미리보기를 확인하세요.
- 직접 마스크를 그리려면 Person의 Load Image에서 Mask Editor로 편집한 뒤 해당 MASK 출력을 추론 노드 `clothing_mask`에 연결합니다. DensePose 연결은 유지합니다.
- 옷 검출 실패 시 `clothing_prompt`를 바꾸거나 threshold를 0.3에서 0.2 정도로 낮춰 봅니다.

## 12GB 메모리 구성

GroundingDINO → SAM ViT-B → DensePose 순으로 전처리하고 종료합니다. 이후 별도 프로세스에서 IDM-VTON을 FP16으로 로드하며, 인물 UNet과 의상 UNet 모두에 순차 CPU 오프로딩을 적용합니다. VAE slicing/tiling을 켭니다. 추론 완료 또는 취소 시 별도 프로세스가 종료되므로 모델 메모리를 반환합니다.

이 구성은 VRAM보다 시스템 RAM과 CPU↔GPU 전송에 더 의존합니다. 다른 대형 모델이나 게임을 동시에 실행하면 메모리 여유가 줄어듭니다. 해상도를 올리면 속도와 메모리 요구가 증가하지만, 768×1024까지는 아래 검증 결과대로 여유가 있습니다.

## 원본 프로젝트

- IDM-VTON 모델: https://huggingface.co/yisol/IDM-VTON
- ComfyUI 구현: https://github.com/TemryL/ComfyUI-IDM-VTON
- GroundingDINO 연동 코드: https://github.com/storyicon/comfyui_segment_anything
- DensePose: 기존 ComfyUI ControlNet Auxiliary Preprocessors 구현
- 예제 인물/의상: https://github.com/yisol/IDM-VTON/tree/main/gradio_demo/example

IDM-VTON 원본 모델/코드는 비상업적 라이선스 조건이 있습니다. 상업적 활용 전 해당 저장소의 라이선스를 확인하세요.

## 검증 상태 (2026-09-18, RTX 5070 12GB)

| 단계 | 결과 |
|---|---|
| SAM 의상 마스크 + DensePose 생성 | 성공, ComfyUI 전처리 노드 약 9초 |
| ComfyUI 노드 등록 / 워크플로 로드 | 누락 노드 없이 정상 |
| 384×512 · 4 steps (스모크, runner 단독) | 성공, 추론 11.5초 / 모델 로딩 포함 29초, 최대 예약 VRAM 1.2GB |
| 384×512 · 20 steps (ComfyUI 전체 워크플로) | 성공, 전처리→추론→저장 약 65초 |
| 768×1024 · 20 steps (runner 단독) | 성공, 추론 43초 / 모델 로딩 포함 60초, 최대 예약 VRAM 2.2GB, 약 1.8s/it |
| 768×1024 · 20 steps (Stability Matrix의 실제 ComfyUI) | 성공, 전처리→추론→저장 약 82초. 재시작 후 새 노드 활성화 및 화면 출력까지 확인 |

- 결과 이미지: 이 작업 폴더의 `outputs\sample_result_384x512.png`, `outputs\sample_result_768x1024.png`
- 측정값은 PyTorch `max_memory_reserved` 기준이며, CPU 오프로딩 때문에 실제 병목은 시스템 RAM과 PCIe 전송입니다. 검증은 시스템 RAM 32GB 환경에서 진행했으며, 그보다 적은 RAM에서는 별도 확인이 필요합니다.
- `IDM-VTON-768x1024.json`을 열고 인물과 의상 이미지만 교체하면 검증한 설정으로 실행됩니다. 완료한 결과는 ComfyUI `output\IDM-VTON`에도 저장되어 있습니다.
- 모든 검증은 공개 예제 인물/의상으로 진행했습니다. 실제 사용 사진에서는 마스크 품질(위 "마스크 확인" 항목)이 결과를 좌우합니다.
