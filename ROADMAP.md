# 🗺️ Project Roadmap — Edge-Optimized Tiny Object Detection using NAS & KD

## 👥 Team
- **Coordinator:** Gulnoza Sabirjonova — 220278@centralasian.uz  
- **Member:** Feruza Khudoyberdiyeva — 220328@centralasian.uz  
- **Repository:** [GitHub - Edge-Optimized Tiny Object Detection](https://github.com/GulnozaS/Edge-optimized-Tiny-Object-Detection-using-Neural-Architecture-Search-and-Knowledge-Distillation.git)

---

## 🧭 Project Timeline (8 Weeks)
| **Week** | **Focus Area** | **Owner(s)** | **Deliverable** | **Due Date** | **Status** |
|-----------|----------------|---------------|------------------|---------------|-------------|
| **W1** | Topic selection, team setup, and one-page outline submission | Gulnoza & Feruza | Project topic + proposal outline | ✅ Completed | ✅ |
| **W2** | Related work collection and dataset verification | Feruza | Related work summary + dataset access verified | ⏳ In progress | 🔄 |
| **W3** | Baseline reproduction (YOLOv5s or similar) on small dataset split | Gulnoza | Baseline mAP/FPS results and logs | ⏳ Planned | ⏳ |
| **W4** | Method design — NAS search space definition + KD setup plan | Both | Proposed method diagram + ablation test plan | ⏳ Planned | ⏳ |
| **W5** | Start full training run on selected dataset | Both | Interim results (mAP@0.5, loss trends) | ⏳ Planned | ⏳ |
| **W6** | Error analysis, model debugging, and risk mitigation | Gulnoza | Failure cases + mitigation actions | ⏳ Planned | ⏳ |
| **W7** | Edge deployment and optimization (Jetson Nano testing) | Feruza | FPS evaluation + optimized model export | ⏳ Planned | ⏳ |
| **W8** | Final evaluation, documentation, and report preparation | Both | Final report + slides + code cleanup | ⏳ Planned | ⏳ |

---

## 📆 Weekly Check-ins
Each week, add 3–6 short update bullets below this section.

### ✅ Week 1 Update
- Topic finalized: *Edge-Optimized Tiny Object Detection using NAS & KD*  
- Repository created and structured  
- Initial README.md and proposal prepared  
- Division of labor established (Gulnoza – modeling, Feruza – data)

### 🔄 Week 2 Update
_(to be filled after week 2)_

---

## ⚙️ Checkpoints Summary
| Milestone | Target Date | Verification |
|------------|--------------|---------------|
| One-page outline submitted | Week 1 | ✅ |
| Dataset access verified | Week 2 | ⏳ |
| Baseline results reproduced | Week 3 | ⏳ |
| Method & metrics finalized | Week 4 | ⏳ |
| Interim results produced | Week 5 | ⏳ |
| Error analysis complete | Week 6 | ⏳ |
| Edge testing complete | Week 7 | ⏳ |
| Final report submitted | Week 8 | ⏳ |

---

## ⚠️ Risks & Mitigation
| Risk | Mitigation Plan |
|------|------------------|
| Dataset too large for available compute | Use a small subset or lightweight dataset (Tiny COCO, VisDrone subset) |
| Model training too slow on Jetson | Pre-train on GPU, then fine-tune on Jetson |
| mAP improvement below 10% | Add data augmentation and refine distillation loss |
| NAS search too computationally heavy | Use lightweight search space and early stopping |

---

## 🔗 Useful References
- **YOLOv5 Baseline:** https://github.com/ultralytics/yolov5  
- **VisDrone Dataset:** https://github.com/VisDrone/VisDrone-Dataset  
- **TinyPerson Dataset:** https://github.com/ucas-vg/TinyBenchmark  
- **Knowledge Distillation Paper:** [Distilling the Knowledge in a Neural Network (Hinton et al., 2015)](https://arxiv.org/abs/1503.02531)

---

### 🏁 Next Steps
- [ ] Add Week 2 updates  
- [ ] Push baseline reproduction logs to `/results`  
- [ ] Track progress weekly with commit tags like `update: week2-checkin`

---

**Last updated:** October 2025  
Maintained by **Gulnoza Sabirjonova (@GulnozaS)**  

