# TG-LoRA タスク概要


<!-- spine:anchor:begin -->
> **Spine anchor**: [TG-LoRA アーキテクチャ設計](../architecture.md)
>
> - parent: `tg-lora/architecture.md`
> - role: `detailed`
> - status: `canonical_child`
<!-- spine:anchor:end -->

**作成日**: 2026-05-21
**最終更新**: 2026-07-04（TASK-0153: torn-write 整合性軸 LOAD 側閉包 — `load_trajectory_delta_artifact` に CheckpointIntegrityError 診断追加 + integrity primitives の zero-dep leaf 化。feedback 指定の optimizer.pt / TrainingState JSON は phantom/`.pt`-not-JSON を検証で確定）
**プロジェクト期間**: 2026-05-21 - 2026-06-19（24日）
**推定工数**: 262.0時間（Phase 1-38: 170.0h + Phase 40-42: 8.5h + Phase 43: 4.0h + Phase 44-47: 13.5h + Phase 48: 3.0h + Phase 49: 3.0h + Phase 50: 5.0h + Phase 51: 12.0h + Phase 52: 2.0h + Phase 53: 0.5h + Phase 54: 2.0h + Phase 55: 2.5h + Phase 56: 3.0h + Phase 57: 7.0h + Phase 58: 4.0h + Phase 59: 3.0h + Phase 60: 3.0h + Phase 61: 3.0h + Phase 62: 8.0h + Phase 63: 2.5h + Phase 64: 1.5h）
**総タスク数**: 133件（131件完了 + 1件Phase 43 GPU依存 + 1件Phase 64 完了）
**要件数**: 259件（REQ-001~217 + REQ-218~231 Phase 56追加 + REQ-232~236 Phase 59追加 + REQ-237~240 Phase 60追加 + REQ-241~250 Phase 57-58追加 + REQ-251~258 Phase 61追加 + REQ-037a 比較スクリプト警告パターン統一）
**次回タスク番号**: TASK-0154

## 関連文書

- **要件定義書**: [📋 requirements.md](../requirements.md)
- **設計文書**: [📐 architecture.md](../architecture.md)
- **データフロー**: [🔄 dataflow.md](../dataflow.md)
- **受け入れ基準**: [✅ acceptance-criteria.md](../acceptance-criteria.md)
- **ユーザストーリー**: [👤 user-stories.md](../user-stories.md)
- **実装分析**: [📝 interview-record.md](../interview-record.md)
- **設計分析**: [📝 design-interview.md](../design-interview.md)

## フェーズ構成

| フェーズ | 期間 | 成果物 | タスク数 | 工数 | ファイル |
|---------|------|--------|----------|------|----------|
| Phase 1 | 3日 | データ・モデル層テスト | 3件 | 5.5h | [TASK-0001~0003](#phase-1-データモデル層テスト追加) |
| Phase 1 | 2日 | 評価・学習層テスト | 2件 | 3.5h | [TASK-0004~0005](#phase-2-評価学習層テスト追加) |
| Phase 1 | 3日 | ドキュメント統合・最終検証 | 2件 | 2.5h | [TASK-0006~0007](#phase-3-ドキュメント統合と最終検証) |
| Phase 2 | 3日 | カバレッジ完成とコード品質向上 | 4件 | 9.5h | [TASK-0008~0011](#phase-4-カバレッジ完成とコード品質向上) |
| Phase 3 | 4日 | 学習ループテスタビリティ・カバレッジ向上 | 4件 | 10.5h | [TASK-0012~0015](#phase-5-学習ループテスタビリティカバレッジ向上) |
| Phase 4 | 3日 | 設定検証と安全性向上 | 4件 | 9.5h | [TASK-0016~0019](#phase-6-設定検証と安全性向上) |
| Phase 5 | 3日 | 運用品質とインフラ改善 | 4件 | 8.5h | [TASK-0020~0023](#phase-7-運用品質とインフラ改善) |
| Phase 6 | 3日 | 再現性とデータ品質向上 | 3件 | 6h | [TASK-0024~0026](#phase-8-再現性とデータ品質向上) |
| Phase 9 | 5日 | GPU学習検証と実験実行 | 5件 | 14h | [TASK-0027~0031](#phase-9-gpu学習検証と実験実行) |
| Phase 10 | 1日 | 外挿安全性・Config文字列検証 | — | — | コミット内実装（REQ-056/057/058） |
| Phase 11 | 1日 | 外挿安全性統合テスト | — | — | コミット内実装（REQ-059/060） |
| Phase 12 | 2日 | 外挿安全性深化 | 2件 | 4.5h | [TASK-0032~0033](#phase-12-外挿安全性深化) |
| Phase 14 | 1日 | 信頼性修正（REQ-061~068） | — | — | コミット内実装（5ファイル57行） |
| Phase 15 | 1日 | Phase 14 検証と最終確認 | 2件 | 3h | [TASK-0034~0035](#phase-15-phase-14-検証と最終確認) |
| Phase 16 | 2日 | 評価メトリクス統一とConfig完全性 | 3件 | 7.5h | [TASK-0036~0038](#phase-16-評価メトリクス統一とconfig完全性) |
| Phase 17 | 2日 | テスト品質とエッジケース補強 | 3件 | 6h | [TASK-0039~0041](#phase-17-テスト品質とエッジケース補強) |
| Phase 18 | 1日 | ドキュメント更新 | 1件 | 1h | [TASK-0042](#phase-18-ドキュメント更新) |
| Phase 19 | 2日 | Perplexity E2E・Property-Based Testing | 3件 | 6h | [TASK-0043~0045](#phase-19-perplexity-e2eproperty-based-testing) |
| Phase 20 | 2日 | テスト堅牢化と完了確認 | 3件 | 4h | [TASK-0046~0048](#phase-20-テスト堅牢化と完了確認) |
| Phase 21 | 1日 | DRYリファクタリングとコード品質 | 2件 | 3h | [TASK-0049~0050](#phase-21-dryリファクタリングとコード品質) |
| Phase 22 | 1日 | 公開API・入力検証・エッジケース強化 | — | 3h | コミット内実装（REQ-073~077） |
| Phase 23 | 1日 | テストカバレッジ強化と堅牢性向上 | 5件 | 8h | [TASK-0051~0055](#phase-23-テストカバレッジ強化と堅牢性向上) |
| Phase 24 | 2日 | MLflow実験管理高度化 | 4件 | 9.5h | [TASK-0056~0059](#phase-24-mlflow実験管理高度化) |
| Phase 25 | 2日 | 実験分析ツール整備 | 3件 | 8h | [TASK-0060~0062](#phase-25-実験分析ツール整備) |
| Phase 26 | 2日 | 本番運用品質 | 3件 | 6h | [TASK-0063~0065](#phase-26-本番運用品質) |
| Phase 29 | 1日 | コード品質・整合性 | 2件 | 3.5h | [TASK-0066~0067](#phase-29-コード品質整合性) |
| Phase 30 | 1日 | OptimizerLifecycleManager E2E検証 | 3件 | 4.5h | [TASK-0068~0070](#phase-30-optimizerlifecyclemanager-e2e検証) |
| Phase 31 | 1日 | Makefile検証と新機能統合テスト | 5件 | 9h | [TASK-0071~0075](#phase-31-makefile検証と新機能統合テスト) |
| Phase 33 | 1日 | AsyncCacheBuilder Acceptance・境界値テスト | 3件 | 4h | [TASK-0076~0078](#phase-33-asynccachebuilder-acceptance境界値テスト) |
| Phase 34 | 1日 | パフォーマンス検証とテスト数同期 | 3件 | 5h | [TASK-0079~0081](#phase-34-パフォーマンス検証とテスト数同期) |
| Phase 35 | 1日 | CI gate・回帰自動検出 | 1件 | 3h | [TASK-0082](#phase-35-ci-gate回帰自動検出) |
| Phase 36 | 1日 | スクリプト健全性・Config検証 | 3件 | 4h | [TASK-0083~0085](#phase-36-スクリプト健全性とconfig検証完全性) |
| Phase 37 | 0.5日 | LR探索統合・propose→training loop配線 | 1件 | 1h | [TASK-0086](#phase-37-lr探索統合とpropose-training-loop配線) |
| Phase 38 | 1日 | Config完全性とテストスイート品質 | 3件 | 4.5h | [TASK-0087~0089](#phase-38-config完全性とテストスイート品質) |
| Phase 40 | 1日 | --resume E2E統合テスト | 1件 | 3h | [TASK-0090](#phase-40---resume-e2e統合テスト) |
| Phase 41 | 1日 | TruthfulQA分析とaccel param実験 | 2件 | 4h | [TASK-0091~0092](#phase-41-truthfulqa分析とaccel-param実験) |
| Phase 42 | 0.5日 | Phase 40-41 ドキュメント更新 | 1件 | 1h | [TASK-0093](#phase-42-ドキュメント更新) |
| Phase 43 | 1日 | Accel param sweep実行と結果分析 | 1件 | 4h | [TASK-0094](#phase-43-sweep実行と結果分析) |
| Phase 44 | 1日 | コンストラクタ検証（学習インフラ） | 2件 | 4h | [TASK-0095~0096](#phase-44-コンストラクタ検証学習インフラ) |
| Phase 45 | 1日 | コンストラクタ検証（ユーティリティ・評価） | 2件 | 4h | [TASK-0097~0098](#phase-45-コンストラクタ検証ユーティリティ評価) |
| Phase 46 | 1日 | フレイキーテスト排除 | 2件 | 3h | [TASK-0099~0100](#phase-46-フレイキーテスト排除) |
| Phase 47 | 0.5日 | ドキュメント更新・検証 | 1件 | 1h | [TASK-0101](#phase-47-ドキュメント更新検証) |
| Phase 48 | 1日 | NaN/Inf バリデーション・ランタイムガード | 1件 | 3h | [TASK-0102](#phase-48-naninf-バリデーションランタイムガード完全化) |
| Phase 49 | 1日 | テスト数同期・CI gate安定性 | 2件 | 3h | [TASK-0103~0104](#phase-49-テスト数同期とci-gate安定性) |
| Phase 50 | 2日 | Stage 2 マルチシード複製・Paper Gate評価 | 2件 | 5h | [TASK-0105~0106](#phase-50-stage-2-マルチシード複製paper-gate評価) |
| Phase 51 | 2日 | Paper Pipeline Stage 3-5自動化 | 4件 | 12h | [TASK-0107~0109, TASK-0111](#phase-51-paper-pipeline-stage-3-5自動化) |
| Phase 52 | 1日 | 論文結果統合 | 1件 | 2h | [TASK-0110](#phase-52-論文結果統合) |
| Phase 56 | 1日 | モデル検査・比較ダッシュボード・ワンショットキャッシュ・コスト分析 | 2件 | 3h | [TASK-0112~0113](#phase-56-モデル検査比較ダッシュボードワンショットキャッシュコスト分析) |
| Phase 57 | 2日 | 論文実験統計分析強化 | 3件 | 7h | [TASK-0114~0116](#phase-57-論文実験統計分析強化) |
| Phase 58 | 1日 | 学習品質モニタリング | 2件 | 4h | [TASK-0117~0118](#phase-58-学習品質モニタリング) |
| Phase 59 | 1日 | 学習軌跡分析・収束予測 | 1件 | 3h | [TASK-0119](#phase-59-学習軌跡分析収束予測早期停止推奨) |
| Phase 60 | 1日 | 軌跡連動適応制御 | 1件 | 3h | [TASK-0120](#phase-60-軌跡連動適応制御) |
| Phase 61 | 1日 | Training Advisor モジュール・CLI | 1件 | 3h | [TASK-0121](#phase-61-training-advisor-モジュールcli) |
| Phase 62 | 1日 | 統合テスト・CI品質強化 | 4件 | 8h | [TASK-0122~0125](#phase-62-統合テストci品質強化) |
| Phase 63 | 0.5日 | 最終仕様整合性と品質確認 | 3件 | 2.5h | [TASK-0126~0128](#phase-63-最終仕様整合性と品質確認) |
| Phase 64 | 0.5日 | parse_warnings E2E検証 | 1件 | 1.5h | [TASK-0129](#phase-64-parse_warnings-e2e検証) |

## タスク番号管理

**使用済みタスク番号**: TASK-0001 ~ TASK-0129
**次回開始番号**: TASK-0130

**テストスイート状況（2026-05-25更新）**: 105テストファイル、2538テストケース（2538 passed, 7 skipped, 0 failed, 0 errors、カバレッジ99%）
**要件カバレッジ（2026-05-25更新）**: 🟡→🔵 更新完了（REQ-301/302/097/100/196, NFR-002, EDGE-003, EDGE-136, REQ-071）、残り🟡はREQ-072（プロセスルール）のみ

## 全体進捗

- [x] Phase 1: データ・モデル層テスト追加 ✅ 2026-05-21
- [x] Phase 1: 評価・学習層テスト追加 ✅ 2026-05-21
- [x] Phase 1: ドキュメント統合と最終検証 ✅ 2026-05-21
- [x] Phase 2: カバレッジ完成とコード品質向上 ✅ 2026-05-21
- [x] Phase 3: 学習ループテスタビリティ・カバレッジ向上 ✅ 2026-05-21
- [x] Phase 4: 設定検証と安全性向上 ✅ 2026-05-21
- [x] Phase 5: 運用品質とインフラ改善 ✅ 2026-05-21
- [x] Phase 6: 再現性とデータ品質向上 ✅ 2026-05-21
- [x] Phase 9: GPU学習検証と実験実行 ✅ 2026-05-21
- [x] Phase 10: 外挿安全性・Config文字列検証 ✅ 2026-05-21
- [x] Phase 11: 外挿安全性統合テスト ✅ 2026-05-21
- [x] Phase 12: 外挿安全性深化 ✅ 2026-05-21
- [x] Phase 14: 信頼性修正 ✅ 2026-05-21
- [x] Phase 15: Phase 14 検証と最終確認 ✅ 2026-05-21
- [x] Phase 16: 評価メトリクス統一とConfig完全性 ✅ 2026-05-22
- [x] Phase 17: テスト品質とエッジケース補強 ✅ 2026-05-22
- [x] Phase 18: ドキュメント更新 ✅ 2026-05-22
- [x] Phase 19: Perplexity E2E・Property-Based Testing ✅ 2026-05-22
- [x] Phase 20: テスト堅牢化と完了確認 ✅ 2026-05-22
- [x] Phase 21: DRYリファクタリングとコード品質 ✅ 2026-05-22
- [x] Phase 23: テストカバレッジ強化と堅牢性向上
- [x] Phase 24: MLflow実験管理高度化 ✅ 2026-05-22
- [x] Phase 25: 実験分析ツール整備 ✅ 2026-05-22
- [x] Phase 26: 本番運用品質 ✅ 2026-05-22
- [x] Phase 28: ActivationCache・決定論的モード・高度Accept/Rollback ✅ 2026-05-23
- [x] Phase 29: コード品質・整合性 ✅ 2026-05-23
- [x] Phase 30: OptimizerLifecycleManager E2E検証
- [x] Phase 31: Makefile検証と新機能統合テスト
- [x] Phase 32a: AsyncCacheBuilder設計文書更新・テスト数更新
- [x] Phase 33: REQ-136~138 acceptance criteria・AsyncCacheBuilder境界値テスト・ドキュメント修正
- [x] Phase 34: パフォーマンス検証とテスト数同期
- [x] Phase 35: CI gate・回帰自動検出（TASK-0082: bench-velocity-ops-ci Makefile target + baseline file） ✅ 2026-05-23
- [x] Phase 36: スクリプト健全性とConfig検証完全性 ✅ 2026-05-24
- [x] Phase 37: LR探索統合とpropose→training loop配線 ✅ 2026-05-24
- [x] Phase 38: Config完全性とテストスイート品質 ✅ 2026-05-24
- [x] Phase 40: --resume E2E統合テスト ✅ 2026-05-24
- [x] Phase 41: TruthfulQA分析とaccel param実験
- [x] Phase 42: Phase 40-41 ドキュメント更新
- [ ] Phase 43: Accel param sweep実行と結果分析
- [x] Phase 48: NaN/Inf バリデーション・ランタイムガード完全化 ✅ 2026-05-25
- [x] Phase 49: テスト数同期とCI gate安定性 ✅ 2026-05-25
- [x] Phase 44: コンストラクタ検証（学習インフラ） ✅ 2026-05-24
- [x] Phase 45: コンストラクタ検証（ユーティリティ・評価） ✅ 2026-05-24
- [x] Phase 46: フレイキーテスト排除 ✅ 2026-05-24
- [x] Phase 47: ドキュメント更新・検証 ✅ 2026-05-24
- [x] Phase 50: Stage 2 マルチシード複製・Paper Gate評価（TASK-0105完了、TASK-0106 GPU依存）
- [x] Phase 51: Paper Pipeline Stage 3-5自動化（TASK-0107/0109/0111完了）
- [x] Phase 52: 論文結果統合（TASK-0110完了）
- [x] Phase 56: モデル検査・比較ダッシュボード・ワンショットキャッシュ・コスト分析 ✅ 2026-05-25
- [x] Phase 57: 論文実験統計分析強化 ✅ 2026-05-25
- [x] Phase 58: 学習品質モニタリング ✅ 2026-05-25
- [x] Phase 59: 学習軌跡分析・収束予測・早期停止推奨 ✅ 2026-05-25
- [x] Phase 60: 軌跡連動適応制御 ✅ 2026-05-25
- [x] Phase 61: Training Advisor モジュール・CLI ✅ 2026-05-25
- [x] Phase 62: 統合テスト・CI品質強化 ✅ 2026-05-25
- [x] Phase 63: 最終仕様整合性と品質確認 ✅ 2026-05-25
- [x] Phase 64: parse_warnings E2E検証 ✅ 2026-05-25

## マイルストーン

- **M1: データ・モデル層テスト完了** (2026-05-21): ✅ テストカバレッジギャップの前半解消
- **M2: 評価・学習層テスト完了** (2026-05-21): ✅ テストカバレッジギャップの完全解消
- **M3: 全受け入れ基準グリーン** (2026-05-21): ✅ acceptance-criteria.md 全テストケースパス
- **M4: テスト可能モジュール100%カバレッジ** (2026-05-21): ✅ run_metrics.py・load_model.py カバレッジ100%達成、6バグ修正完了
- **M5: 学習ループ80%+カバレッジ** (2026-05-21): ✅ 純粋関数抽出・モック統合テスト・baselineテスト完了、総合カバレッジ98%到達
- **M6: 運用品質向上** (2026-05-21): ✅ TASK-0020（docs/llm-wiki untracking）完了
- **M7: 再現性・データ品質** (2026-05-21): ✅ Phase 6 完了
- **M8: GPU学習検証完了** (2026-05-21): ✅ Phase 9 完了
- **M9: 外挿安全性完了** (2026-05-21): ✅ Phase 10/11 完了（REQ-056~060）
- **M10: 安全性深化完了** (2026-05-21): ✅ Phase 12 完了（TASK-0032/0033）
- **M11: 信頼性修正完了** (2026-05-21): ✅ Phase 14 完了（REQ-061~068, EDGE-126~134, 5ファイル57行変更）
- **M12: 最終検証完了** (2026-05-21): ✅ Phase 15 完了（TASK-0034/0035、773テスト全パス確認）
- **M13: メトリクス・Config統一完了** (2026-05-22): ✅ Phase 16 完了（EvalLossResult統合、Config完全性、MLflow一貫性）
- **M14: テスト品質補強完了** (2026-05-22): ✅ Phase 17 完了（temperature統合テスト、perplexity E2E、スタブ補完）
- **M15: Perplexity E2E・Property Testing完了** (2026-05-22): ✅ Phase 19 完了（perplexity E2E、trainer パリティ、accept()プロパティテスト）
- **M16: テストスイート全通過完了** (2026-05-22): ✅ Phase 20 完了（863 passed, 9 skipped, 0 failed, 0 errors）
- **M17: DRYリファクタリング完了** (2026-05-22): ✅ Phase 21 完了（InfiniteBatchIterator・StrategyList・CheckpointHelper抽出）
- **M18: テストカバレッジ強化完了** (2026-05-22): ✅ Phase 23 完了（checkpoint readback・batch iterator エッジケース・warning log・rollback E2E。891テスト全パス）
- **M19: 実験分析ツール整備完了** (2026-05-22): ✅ Phase 25 完了（query API・multi-run dashboard・visualization。1009テスト全パス、カバレッジ99%）
- **M20: 本番運用品質完了** (2026-05-22): ✅ Phase 26 完了（TrainingState・fault recovery・diagnostics・API reference。1122テスト全パス）
- **M21: 全タスク完了** (2026-05-22): ✅ 全65タスク完了（Phase 24-26 ドキュメント更新完了、テストスイート全通過確認）
- **M22: Phase 28機能統合完了** (2026-05-23): ✅ ActivationCache・決定論的モード・移動平均ベースライン・soft accept・K-step中間ロールバック・confident-skip（REQ-110~118）
- **M23: Phase 29 コード品質完了** (2026-05-23): ✅ lint/format クリーンアップ・overview整合性更新（ruff 0エラー・1139テスト全通過）
- **M24: Phase 30 OptimizerLifecycleManager E2E検証完了** (2026-05-23): OptimizerLifecycleManager E2Eスモークテスト・ベンチマークテスト・ドキュメント更新
- **M25: Phase 31 Makefile検証と新機能統合完了** (2026-05-23): Makefileターゲット検証・trainable_lora_scope統合テスト・prefix_feature_cache拡張テスト・実験config検証
- **M26: Phase 32a 全タスク完了** (2026-05-23): AsyncCacheBuilder設計文書更新・テスト数1289更新・全75タスク完了
- **M27: Phase 33 AsyncCacheBuilder Acceptance完了** (2026-05-23): REQ-136~138 acceptance criteria追加・AsyncCacheBuilder境界値テスト3件追加・1290テスト全パス
- **M28: Phase 34 パフォーマンス検証完了** (2026-05-23): in-place tensor ops検証・マイクロベンチマーク・テスト数1314同期
- **M29: Phase 36 スクリプト健全性・Config検証完了** (2026-05-24): インポート健全性テスト・Config検証統合・テスト数1393同期・全85タスク完了
- **M30: Phase 37 LR探索統合完了** (2026-05-24): LR探索Config→Controller→Optimizer配線検証・REQ-150~152完了
- **M31: Phase 38 Config完全性・品質完了** (2026-05-24): LR探索Config明示化・テスト警告解消・ドキュメント同期・89タスク全完了
- **M32: Phase 40 --resume E2E完了** (2026-05-24): --resume E2E統合テスト（save→interrupt→resume→loss継続検証）・1597テスト全パス
- **M33: Phase 57 論文実験統計分析完了** (2026-05-26): マルチシード統計モジュール・論文エクスポート・感度分析
- **M34: Phase 58 学習品質モニタリング完了** (2026-05-26): サイクル健全性モニター・クロス構成コンパレータ
- **M35: Phase 59 学習軌跡分析完了** (2026-05-25): TrajectoryAnalyzer・CLI・収束予測・早期停止推奨
- **M36: Phase 60 軌跡連動適応制御完了** (2026-05-25): TrajectoryController・異常検知適応・収束駆動パラメータ調整
- **M37: Phase 61 Training Advisor完了** (2026-05-25): TrainingAdvisor・AdvisoryReport・advise_training.py CLI・統合監視アドバイザ
- **M38: Phase 63 最終仕様整合性完了** (2026-05-25): 6件🟡→🔵更新・trap handler追加・overview最新化・2524テスト全パス
- **M39: Phase 64 parse_warnings E2E検証完了** (2026-05-25): corrupt JSONL E2Eテスト10件追加・EDGE-003/136/REQ-071 🟡→🔵更新・2538テスト全パス

---

## Phase 1: データ・モデル層テスト追加

**期間**: 3日
**目標**: データパイプライン・モデル管理モジュールのユニットテスト追加
**成果物**: 4テストファイル（test_build_seed_dataset, test_filter_dataset, test_dedup, test_provenance, test_lora_utils, test_load_model）

### タスク一覧

- [x] [TASK-0001: build_seed_dataset ユニットテスト追加](TASK-0001.md) - 1.5h (TDD) 🔵 ✅
- [x] [TASK-0002: filter_dataset, dedup, provenance ユニットテスト追加](TASK-0002.md) - 2h (TDD) 🔵 ✅
- [x] [TASK-0003: load_model, lora_utils ユニットテスト追加](TASK-0003.md) - 2h (TDD) 🔵🟡 ✅

### 依存関係

```
TASK-0001 ──┐
TASK-0002 ──┼── TASK-0006
TASK-0003 ──┘
```

（Phase 1内のタスクは並行実行可能）

---

## Phase 2: 評価・学習層テスト追加

**期間**: 2日
**目標**: 評価モジュール・学習ループプリミティブのユニットテスト追加
**成果物**: 3テストファイル（test_eval_loss, test_eval_modules, test_trainer_loop）

### タスク一覧

- [x] [TASK-0004: eval_loss, eval_task, eval_format ユニットテスト追加](TASK-0004.md) - 2h (TDD) 🔵 ✅
- [x] [TASK-0005: trainer_loop ユニットテスト追加](TASK-0005.md) - 1.5h (TDD) 🔵 ✅

### 依存関係

```
TASK-0004 ──┐
            ├── TASK-0006
TASK-0005 ──┘
```

（Phase 2内のタスクは並行実行可能）

---

## Phase 3: ドキュメント統合と最終検証

**期間**: 3日
**目標**: AGENTS.md同期、MLflow実態確認、受け入れ基準の全テストケース検証
**成果物**: 更新済みAGENTS.md、テストカバレッジレポート

### タスク一覧

- [x] [TASK-0006: AGENTS.md と要件定義の同期](TASK-0006.md) - 1h (DIRECT) 🔵🟡 ✅
- [x] [TASK-0007: 受け入れ基準の全テストケース検証](TASK-0007.md) - 1.5h (DIRECT) 🔵 ✅

### 依存関係

```
TASK-0001 ~ TASK-0005 → TASK-0006 → TASK-0007
```

---

## Phase 4: カバレッジ完成とコード品質向上

**期間**: 3日
**目標**: 残存するテスト可能モジュールのカバレッジを100%に完成させ、ソースコード精査でバグを発見・修正する
**成果物**: 拡張テストスイート、バグ修正、更新済みドキュメント

### タスク一覧

- [x] [TASK-0008: run_metrics.py GPUパス カバレッジ完成](TASK-0008.md) - 1.5h (TDD) 🔵 ✅
- [x] [TASK-0009: load_model.py モックベースユニットテスト追加](TASK-0009.md) - 3h (TDD) 🔵 ✅
- [x] [TASK-0010: コアモジュール ソースコード精査とバグ修正](TASK-0010.md) - 4h (TDD) 🟡 ✅
- [x] [TASK-0011: Phase 2 受け入れ基準・ドキュメント更新](TASK-0011.md) - 1h (DIRECT) 🔵 ✅

### 依存関係

```
TASK-0008 ──┐
TASK-0009 ──┼── TASK-0011
TASK-0010 ──┘
```

（TASK-0008, 0009, 0010 は並行実行可能）

---

## 信頼性レベルサマリー

### 全タスク統計

- **総タスク数**: 123件（117件完了、8件未着手）
- 🔵 **青信号**: 96件 (78%)
- 🟡 **黄信号**: 18件 (15%)
- 🔴 **赤信号**: 0件 (0%)

### フェーズ別信頼性

| フェーズ | 🔵 青 | 🟡 黄 | 🔴 赤 | 合計 |
|---------|-------|-------|-------|------|
| Phase 1 (データ・モデル) | 2 | 1 | 0 | 3 |
| Phase 1 (評価・学習) | 2 | 0 | 0 | 2 |
| Phase 1 (ドキュメント) | 1 | 1 | 0 | 2 |
| Phase 2 (カバレッジ・品質) | 3 | 1 | 0 | 4 |
| Phase 3 (学習ループ) | 3 | 1 | 0 | 4 |
| Phase 4 (設定検証・安全性) | 4 | 0 | 0 | 4 |
| Phase 5 (運用品質・インフラ) | 2 | 2 | 0 | 4 |
| Phase 6 (再現性・データ品質) | 1 | 2 | 0 | 3 |
| Phase 9 (GPU学習検証) | 5 | 0 | 0 | 5 |
| Phase 10-11 (外挿安全性) | — | — | — | 完了 |
| Phase 12 (安全性深化) | 2 | 0 | 0 | 2 |
| Phase 14 (信頼性修正) | — | — | — | 完了 |
| Phase 15 (検証) | 2 | 0 | 0 | 2 |
| Phase 16 (メトリクス・Config) | 2 | 1 | 0 | 3 |
| Phase 17 (テスト品質) | 1 | 2 | 0 | 3 |
| Phase 18 (ドキュメント) | 1 | 0 | 0 | 1 |
| Phase 19 (Perplexity/Property) | 2 | 1 | 0 | 3 |
| Phase 20 (テスト堅牢化) | 2 | 1 | 0 | 3 |
| Phase 21 (DRYリファクタリング) | 2 | 0 | 0 | 2 |
| Phase 22 (公開API・入力検証) | — | — | — | 完了 |
| Phase 23 (テストカバレッジ強化) | 4 | 1 | 0 | 5 |
| Phase 24 (MLflow実験管理) | 3 | 1 | 0 | 4 |
| Phase 25 (実験分析ツール) | 2 | 1 | 0 | 3 |
| Phase 26 (本番運用品質) | 3 | 0 | 0 | 3 |
| Phase 29 (コード品質・整合性) | 2 | 0 | 0 | 2 |
| Phase 30 (E2E検証) | 3 | 0 | 0 | 3 |
| Phase 31 (Makefile検証・新機能統合) | 5 | 0 | 0 | 5 |
| Phase 33 (AsyncCacheBuilder Acceptance) | 3 | 0 | 0 | 3 |
| Phase 34 (パフォーマンス検証) | 3 | 0 | 0 | 3 |
| Phase 36 (スクリプト健全性) | 3 | 0 | 0 | 3 |
| Phase 38 (Config完全性・品質) | 3 | 0 | 0 | 3 |
| Phase 40 (--resume E2E) | 1 | 0 | 0 | 1 |
| Phase 41 (TruthfulQA・accel) | 2 | 1 | 0 | 3 |
| Phase 42 (ドキュメント更新) | 1 | 0 | 0 | 1 |
| Phase 44 (コンストラクタ検証・学習) | 2 | 0 | 0 | 2 |
| Phase 45 (コンストラクタ検証・評価) | 2 | 0 | 0 | 2 |
| Phase 46 (フレイキーテスト排除) | 2 | 0 | 0 | 2 |
| Phase 47 (ドキュメント更新) | 1 | 0 | 0 | 1 |
| Phase 48 (NaN/Inf バリデーション) | 1 | 0 | 0 | 1 |
| Phase 49 (テスト数同期・CI gate) | 2 | 0 | 0 | 2 |
| Phase 50 (Stage 2 マルチシード・Gate評価) | 2 | 0 | 0 | 2 |
| Phase 51 (Paper Pipeline Stage 3-5自動化) | 4 | 0 | 0 | 4 |
| Phase 52 (論文結果統合) | 1 | 0 | 0 | 1 |
| Phase 57 (論文実験統計分析) | 0 | 3 | 0 | 3 |
| Phase 58 (学習品質モニタリング) | 0 | 2 | 0 | 2 |
| Phase 60 (軌跡連動適応制御) | 0 | 1 | 0 | 1 |
| Phase 61 (Training Advisor) | 0 | 1 | 0 | 1 |
| Phase 62 (統合テスト・CI品質強化) | 4 | 0 | 0 | 4 |

**品質評価**: 高品質

## クリティカルパス

```
TASK-0001 → TASK-0006 → TASK-0007 → (Phase 1 完了)
TASK-0010 → TASK-0011 (Phase 2 最長パス)
TASK-0012 → TASK-0013 → TASK-0015 (Phase 3 最長パス)
TASK-0031 → TASK-0032 → TASK-0033 → TASK-0034 → TASK-0035 (Phase 9~15 最長パス)
TASK-0112 → TASK-0114 → TASK-0115 → TASK-0116 → TASK-0117 → TASK-0118 (Phase 56~58 最長パス)
```

**クリティカルパス工数**: Phase 1 = 4時間, Phase 2 = 5時間, Phase 3 = 8時間
**並行作業可能工数**: Phase 1 = 7.5時間, Phase 2 = 8.5時間, Phase 3 = 2.5時間

---

## Phase 5: 学習ループテスタビリティ・カバレッジ向上

**期間**: 4日
**目標**: train_tg_lora.py・train_baseline_qlora.py のテストカバレッジを大幅に向上させ、全ソースモジュールのテスト可能性を完成させる
**成果物**: 純粋関数抽出、モック統合テスト、baseline テスト、更新済みドキュメント

### タスク一覧

- [x] [TASK-0012: train_tg_lora.py 純粋関数抽出](TASK-0012.md) - 3h (TDD) 🔵 ✅
- [x] [TASK-0013: モックベース学習ループ統合テスト](TASK-0013.md) - 4h (TDD) 🔵 ✅
- [x] [TASK-0014: train_baseline_qlora.py モックテストカバレッジ](TASK-0014.md) - 2.5h (TDD) 🟡 ✅
- [x] [TASK-0015: Phase 3 受け入れ基準・ドキュメント更新](TASK-0015.md) - 1h (DIRECT) 🔵 ✅

### 依存関係

```
TASK-0012 → TASK-0013 ──┐
TASK-0014 ──────────────┼── TASK-0015
```

（TASK-0012 と TASK-0014 は並行実行可能）

---

## Phase 6: 設定検証と安全性向上

**期間**: 3日
**目標**: Pydantic設定スキーマによる設定検証、CLIエントリポイントのモックテスト（カバレッジ100%達成）、学習開始前バリデーションを追加する
**成果物**: 設定スキーマ、CLIモックテスト、preflightバリデーション、更新済みドキュメント

### タスク一覧

- [x] [TASK-0016: Pydantic設定スキーマによる設定検証](TASK-0016.md) - 3h (TDD) 🔵 ✅
- [x] [TASK-0017: CLIエントリポイントGPUモックテスト](TASK-0017.md) - 2.5h (TDD) 🔵 ✅
- [x] [TASK-0018: 学習開始前バリデーションと設定スキーマ統合](TASK-0018.md) - 3h (TDD) 🔵 ✅
- [x] [TASK-0019: Phase 4 受け入れ基準・ドキュメント更新](TASK-0019.md) - 1h (DIRECT) 🔵 ✅

### 依存関係

```
TASK-0016 ──┬── TASK-0018 ──┐
TASK-0017 ──┘               ├── TASK-0019
```

（TASK-0016 と TASK-0017 は並行実行可能）

## カバレッジ推移

| 時点 | テスト数 | カバレッジ | 備考 |
|------|----------|-----------|------|
| Phase 1 開始時 | 0 | 0% | テスト未整備 |
| Phase 1 完了時 | 123 | 46% | テスト可能モジュール大部分カバー |
| Phase 1 最終（TASK-0007） | 234 | 72% | テスト可能モジュールほぼ100% |
| Phase 2 完了時（TASK-0011） | 262 | 76% | テスト可能モジュール100%、6バグ修正 |
| Phase 2+CycleState/DeltaTracker | 321 | 76% | CycleState(19), DeltaTracker(21), Integration(7), _InfiniteBatchIterator テスト追加 |
| Phase 3 完了時（TASK-0015） | 427 | 98% | 純粋関数(32)・モック統合テスト(45)・baselineテスト(35)追加 |
| Phase 4 完了時（TASK-0019） | 476 | 99% | 設定スキーマ(28)・CLIモック(12)・preflightテスト(8)追加、main()にpreflight統合 |
| Phase 5+6 完了時 | 562 | 99% | MLflow統合・CI/CD・Docker・データスキーマ・可視化強化・Velocity異常検出統合テスト追加 |
| Phase 9 直前（テスト増分） | 575 | 99% | lr境界テスト(3)・収束適応テスト(2)・adaptive LRスモーク(3)・学習ループ統合(5)追加 |
| Phase 14 完了時 | 764 | 99% | config_schema(10)・delta_tracker(5)・metrics(3)・rollback_manager(5)・extrapolation_safety_direct(4テスト更新)追加 |
| Phase 15 TASK-0034完了時 | 773 | 99% | 全テストスイート検証完了・773テストパス確認・リグレッションなし |
| Phase 16-17 完了時 | 853 | 99% | EvalLossResult統合・Config完全性・MLflow一貫性・temperatureテスト・perplexity E2E・スタブ補完（+80テスト） |
| Phase 19 完了時（予定） | ~863 | 99% | Perplexity E2E統合テスト・Trainerパリティテスト・accept()プロパティベーステスト（+10テスト予定） |
| Phase 20 完了時（予定） | ~873 | 99% | GPUテストOOM保護・全テスト通過確認（+0~10テスト予定） |
| Phase 20 完了時（実績） | 863 | 99% | 863 passed, 9 skipped, 0 failed, 0 errors — 全テスト通過確認完了 |
| Phase 21-22 完了時 | 867 | 99% | 867 passed, 9 skipped — 公開API・入力検証・エッジケース強化（+4テスト） |
| Phase 23 完了時（実績） | 891 | 99% | 891 passed, 9 skipped — checkpoint readback・batch iterator エッジケース・warning log・rollback E2E（+24テスト） |
| Phase 25 完了時（実績） | 1009 | 99% | 1009 passed, 9 skipped — MLflow retry・TG-LoRA metrics・query API・multi-run dashboard・visualization（+118テスト） |
| Phase 26 完了時（実績） | 1122 | 99% | 1122 passed, 7 skipped — TrainingState・ControllerState・fault recovery・diagnostics・API exports・artifact logging（+113テスト） |
| Phase 28 完了時（実績） | 1138 | 99% | 1138 passed, 7 skipped — ActivationCache・決定論的モード・移動平均ベースライン・soft accept・K-step中間ロールバック・confident-skip（+16テスト） |
| Phase 29 完了時（実績） | 1139 | 99% | 1139 passed, 7 skipped — ruff lint/format クリーンアップ（99エラー修正・52ファイルフォーマット）・overview整合性更新（+1テスト） |
| Phase 31 完了時（実績） | 1231 | 99% | 1231 passed, 7 skipped — Makefileターゲット検証・trainable_lora_scope統合テスト・prefix_feature_cache拡張テスト・実験config検証（+92テスト） |
| Phase 32 完了時（実績） | 1289 | 99% | 1289 passed, 7 skipped — AsyncCacheBuilder unit tests (4) + config validation tests (2) + REQ-128~135 prefix_feature_cache robustness tests (52)（+58テスト） |
| Phase 33 完了時（実績） | 1290 | 99% | 1290 passed, 9 skipped — REQ-136~138 acceptance criteria + AsyncCacheBuilder edge case tests (get_result nonexistent, force_rebuild, partial failure)（+3テスト） |
| Phase 34 開始時（Phase 33後コミット） | 1314 | 99% | 1314 passed, 7 skipped — REQ-139 integration tests (5) + model_factory injection smoke tests (10) + REQ-140/142 acceptance tests + conftest extraction + REQ-141 velocity warnings + in-place tensor ops（+24テスト） |
| Phase 35 完了時（実績） | 1354 | 99% | 1354 collected, 1347 passed, 7 skipped — bench-velocity-ops-ci CI gate・baseline regression detection・TASK-0079/0080完了（+40テスト） |
| Phase 36 完了時（実績） | 1393 | 99% | 1393 collected, 1386 passed, 7 skipped — スクリプト健全性テスト・Config検証統合・overview同期（+39テスト） |
| Phase 37 完了時（実績） | 1442 | 99% | 1442 passed, 9 skipped — LR探索統合・propose→training loop配線（+49テスト） |
| Phase 37+追加テスト時（実績） | 1451 | 99% | 1451 collected, 1442 passed, 9 skipped — 追加テスト（+9テスト） |
| Phase 39現在（実績） | 1594 | 99% | 1594 collected, 80テストファイル — --resume + accel observabilityテスト追加 |
| Phase 38 完了時（実績） | 1527 | 99% | 1527 collected, 1518 passed, 9 skipped — LR探索Config明示化・テスト警告解消・ドキュメント同期（+76テスト） |
| Phase 40 完了時（実績） | 1597 | 99% | 1597 collected, 1588 passed, 9 skipped — --resume E2E統合テスト（test_resume_e2e.py 3テスト）（+3テスト） |
| Phase 44-47 完了時（実績） | 1922 | 99% | 1922 passed, 7 skipped — コンストラクタ検証・フレイキーテスト排除・ベースライン更新（+325テスト） |
| Phase 48 完了時（実績） | 1999 | 99% | 1999 passed, 7 skipped — NaN/Inf バリデーション修正・ランタイムガード・lr_log_sigma検証・スクリプト防御（+77テスト、内11新規+66探索prob監査再計算） |
| Phase 49 開始時（実績） | 2032 | 99% | 2032 collected, 2025 passed, 7 skipped — compare_paper_memory_modes 21テスト追加・その他テスト追加（+33テスト・94テストファイル） |
| Phase 52 完了時（実績） | 2140 | 99% | 2140 passed, 9 skipped — consolidate_paper_results.py（26テスト）・TASK-0110完了（+26テスト・95テストファイル） |
| Phase 55 運用スクリプトテスト（実績） | 2192 | 99% | 2192 passed, 7 skipped — operational script smoke tests（34テスト）・TASK-0104/0111完了（+34テスト・96テストファイル） |
| Phase 57-58 完了時（実績） | 2330 | 99% | 2330 passed, 7 skipped — paper export（14テスト）・sensitivity analysis（14テスト）・cycle monitor（22テスト）・experiment comparator（15テスト）（+138テスト・100テストファイル） |
| Phase 59 完了時（実績） | 2399 | 99% | 2399 passed, 7 skipped — trajectory analysis（69テスト）（+69テスト・101テストファイル） |
| Phase 60 完了時（実績） | 2434 | 99% | 2434 passed, 7 skipped — trajectory-informed adaptive control（+35テスト・103テストファイル） |
| Phase 61 完了時（実績） | 2469 | 99% | 2469 passed, 7 skipped — training advisor（+35テスト・104テストファイル） |

### Phase 2 カバレッジ改善詳細

| モジュール | Phase 1終了時 | Phase 2終了時 | 変化 |
|-----------|-------------|-------------|------|
| src/utils/run_metrics.py | 85% | 100% | +15% |
| src/model/load_model.py | 35% | 100% | +65% |
| src/tg_lora/extrapolator.py | 95% | 100% | +5% |
| src/tg_lora/lora_state.py | 90% | 100% | +10% |
| src/tg_lora/velocity.py | 97% | 100% | +3% |
| src/eval/eval_format.py | 90% | 100% | +10% |
| src/eval/eval_task.py | 90% | 100% | +10% |
| その他全モジュール | — | 100% | — |
| **総合カバレッジ** | **72%** | **76%** | **+4%** |

**注**: 総合カバレッジ76%は、GPU専用モジュール（train_baseline_qlora.py, train_tg_lora.py = 0%/149行）が分母に含まれるため。テスト可能モジュール（GPU依存なし）は全て100%に到達。

### Phase 3 カバレッジ改善詳細

| モジュール | Phase 2終了時 | Phase 3終了時 | 変化 |
|-----------|-------------|-------------|------|
| src/training/train_tg_lora.py | 0% | 94% | +94% |
| src/training/train_baseline_qlora.py | 0% | 85% | +85% |
| **総合カバレッジ** | **76%** | **98%** | **+22%** |

**注**: train_tg_lora.pyの残り6%はmain()のCLI起動部分、train_baseline_qlora.pyの残り15%はmain()とデータセット読み込み部分。いずれもGPU環境依存。

### Phase 4 カバレッジ実績

| モジュール | Phase 3終了時 | Phase 4実績 | 変化 |
|-----------|-------------|-----------|------|
| src/training/train_tg_lora.py | 94% | 99% | +5% |
| src/training/train_baseline_qlora.py | 85% | 96% | +11% |
| src/training/config_schema.py | N/A | 100% | 新規 |
| src/training/preflight.py | N/A | 96% | 新規 |
| **総合カバレッジ** | **98%** | **99%** | **+1%** |

---

## Phase 7: 運用品質とインフラ改善

**期間**: 3日
**目標**: AI Hub フィードバック対応、MLflow 統合、CI/CD 自動化により運用品質を向上させる
**成果物**: wiki untracking、強化テスト、MLflow ロギングモジュール、GitHub Actions パイプライン

### タスク一覧

- [x] [TASK-0020: docs/llm-wiki git tracking 解除とクリーンアップ](TASK-0020.md) - 0.5h (DIRECT) 🔵 ✅
- [x] [TASK-0021: _InfiniteBatchIterator StopIteration テスト強化](TASK-0021.md) - 1h (TDD) 🔵 ✅
- [x] [TASK-0022: MLflow 実験ロギング統合](TASK-0022.md) - 4h (TDD) 🟡 ✅
- [x] [TASK-0023: GitHub Actions CI/CD パイプライン構築](TASK-0023.md) - 3h (DIRECT) 🟡 ✅

### 依存関係

```
TASK-0020 ──┬── TASK-0022 ──┐
TASK-0021 ──┤               ├── TASK-0024
            └── TASK-0023 ──┘
```

（TASK-0020 と TASK-0021 は並行実行可能）

---

## Phase 8: 再現性とデータ品質向上

**期間**: 3日
**目標**: Docker 環境による再現性確保、データバリデーション、レポート可視化強化
**成果物**: Dockerfile、データスキーマバリデーション、強化比較レポート

### タスク一覧

- [x] [TASK-0024: Docker 開発環境構築](TASK-0024.md) - 2h (DIRECT) 🟡 ✅
- [x] [TASK-0025: データスキーマバリデーション追加](TASK-0025.md) - 2h (TDD) 🔵 ✅
- [x] [TASK-0026: 比較レポート可視化強化](TASK-0026.md) - 2h (TDD) 🔵 ✅

### 依存関係

```
TASK-0023 → TASK-0024
TASK-0022 → TASK-0025 → TASK-0026
```

---

## Phase 9: GPU学習検証と実験実行

**期間**: 5日
**目標**: GPU環境でTG-LoRA学習を実際に実行し、アルゴリズムの動作と効率性を実験的に検証する
**成果物**: 学習メトリクス、比較レポート、実験根拠に基づくドキュメント更新

### タスク一覧

- [x] [TASK-0027: GPU学習環境準備とモデル読み込み検証](TASK-0027.md) - 2h (DIRECT) 🔵 ✅
- [x] [TASK-0028: TG-LoRA 10サイクル学習スモークテスト](TASK-0028.md) - 3h (TDD) 🔵 ✅
- [x] [TASK-0029: ベースラインQLoRA学習実行](TASK-0029.md) - 3h (TDD) 🔵 ✅
- [x] [TASK-0030: 公正比較実験と結果分析](TASK-0030.md) - 4h (TDD) 🔵 ✅
- [x] [TASK-0031: Phase 9 受け入れ基準・ドキュメント更新](TASK-0031.md) - 2h (DIRECT) 🔵 ✅

### 依存関係

```
TASK-0027 ──┬── TASK-0028 ──┬── TASK-0030 ── TASK-0031
            └── TASK-0029 ──┘
```

（TASK-0028 と TASK-0029 は並行実行可能）

---

## Phase 10: 外挿安全性・Config文字列検証

**期間**: 1日
**目標**: 外挿後のNaN/Inf検出機構と設定スキーマのdtype列挙検証を追加する
**成果物**: check_lora_params_finite関数、dtype Literal enum、単体テスト

### 成果（コミット内実装、正式タスクなし）

- REQ-056: 外挿後パラメータ有限性検証（check_lora_params_finite） ✅
- REQ-057: trainer間数値安全性カバレッジ一致 ✅
- REQ-058: dtype/bnb_4bit_compute_dtype Literal enum検証 ✅

### 関連コミット

- `9754565 feat(training): add post-extrapolation NaN/Inf safety check and dtype Literal enum`

---

## Phase 11: 外挿安全性統合テスト

**期間**: 1日
**目標**: 外挿→NaN検出→ロールバック→penalize→cycle_state.record_cycleの完全な回復フローを統合テストで検証する
**成果物**: test_extrapolation_safety_integration.py（REQ-059/060）

### 成果（コミット内実装、正式タスクなし）

- REQ-059: 非有限パラメータ回復フロー統合テスト（4テストケース） ✅
- REQ-060: 回復フロー副作用検証（4テストケース） ✅
- 回復パスコンポーネント直接検証（6テストケース） ✅

### 関連コミット

- `66a37b3 test(safety): add REQ-059/060 integration tests for extrapolation NaN recovery flow`
- `d0f63ce docs(specs): add REQ-059/060 integration test gaps for extrapolation safety recovery flow`

---

## Phase 12: 外挿安全性深化

**期間**: 2日
**目標**: 既存のモックベース統合テストを補完し、実外挿コードによるNaN検出と多様化障害サイクルの統合検証を追加する
**成果物**: 実外挿NaN検出テスト、多様化障害サイクルテスト

### タスク一覧

- [x] [TASK-0032: 実外挿コードによるNaN検出統合テスト](TASK-0032.md) - 2h (TDD) 🔵 ✅
- [x] [TASK-0033: 多様化障害サイクル統合テスト](TASK-0033.md) - 2.5h (TDD) 🔵 ✅

### 依存関係

```
TASK-0031 ── TASK-0032 ── TASK-0033
```

---

## Phase 14: 信頼性修正

**期間**: 1日
**目標**: 設定安全性・数値信頼性の5件の修正を適用する
**成果物**: 5ファイル57行変更、27新規テストメソッド

### 成果（コミット内実装）

- REQ-061: config_schema全11モデル extra='forbid'（RISK-0015/0016） ✅
- REQ-062: 非mapping YAML拒否 ✅
- REQ-063: cap_update非有限ゼロ返却 ✅
- REQ-064: ロールバックスナップショットNaN/Infサニタイズ（RISK-0074） ✅
- REQ-065: ロールバック履歴max_history制限（RISK-0074） ✅
- REQ-066: metrics.cosine_similarityキー不一致安全処理 ✅
- REQ-067: DeltaTracker._compute_stats非有限スキップ ✅
- REQ-068: DeltaTracker norm_history非有限ガード ✅

### 関連コミット

- `81ee464 fix(metrics,delta_tracker): prevent KeyError on key mismatch and sanitize NaN/Inf in stats`
- `7720c98 fix(rollback): sanitize NaN/Inf in snapshots and bound history size (RISK-0074)`
- `f03932a fix(extrapolator): prevent NaN corruption from infinite velocity overflow`
- `e80840e docs(specs): update architecture and dataflow docs for Phase 14 reliability fixes`
- `cc8e19f docs(specs): reflect Phase 14 reliability fixes in requirements (REQ-061~068, EDGE-126~134)`

---

## Phase 15: Phase 14 検証と最終確認

**期間**: 1日
**目標**: Phase 14の信頼性修正後の全テストスイート検証とドキュメント最終更新
**成果物**: 検証済みテストスイート、更新済みドキュメント

### タスク一覧

- [x] [TASK-0034: Phase 14 信頼性修正の全テストスイート検証とリグレッション確認](TASK-0034.md) - 2h (TDD) 🔵 ✅
- [x] [TASK-0035: Phase 15 ドキュメント・受け入れ基準更新](TASK-0035.md) - 1h (DIRECT) 🔵 ✅

### 依存関係

```
TASK-0033 ── TASK-0034 ── TASK-0035
```

---

## 次のステップ

Phase 16-18 のタスクを実装するには:

- 全タスク順番に実装: `/tsumiki:kairo-implement`
- 特定タスクを実装: `/tsumiki:kairo-implement TASK-0036`

---

## Phase 16: 評価メトリクス統一とConfig完全性

**期間**: 2日
**目標**: ベースライン学習にEvalLossResult（perplexity/min/max）を統合し、Config項目の完全性を確保し、MLflowロギングの一貫性を向上させる
**成果物**: 評価メトリクス統一、Config完全性、MLflow一貫性

### タスク一覧

- [x] [TASK-0036: ベースライン学習にEvalLossResult統合](TASK-0036.md) - 3h (TDD) 🔵 ✅
- [x] [TASK-0037: 未使用Config項目の整理と早期停止パラメータ露出](TASK-0037.md) - 2.5h (TDD) 🔵🟡 ✅
- [x] [TASK-0038: MLflow ロギングのベースライン/TG-LoRA間一貫性確保](TASK-0038.md) - 2h (TDD) 🔵 ✅

### 依存関係

```
TASK-0036 ──── TASK-0038
TASK-0037
```

（TASK-0036 と TASK-0037 は並行実行可能）

---

## Phase 17: テスト品質とエッジケース補強

**期間**: 2日
**目標**: Layer Sampler temperatureパラメータの統合テスト、RunMetrics perplexity出力、空テストスタブの補完
**成果物**: テスト品質向上、エッジケースカバレッジ拡大

### タスク一覧

- [x] [TASK-0039: Layer Sampler temperatureパラメータ統合テスト](TASK-0039.md) - 2h (TDD) 🔵🟡 ✅
- [x] [TASK-0040: RunMetrics perplexity出力とEvalLossResult E2Eテスト](TASK-0040.md) - 2h (TDD) 🔵 ✅
- [x] [TASK-0041: 空テストスタブ補完とエッジケースカバレッジ向上](TASK-0041.md) - 2h (TDD) 🟡 ✅

### 依存関係

```
TASK-0037 ── TASK-0039
TASK-0036 ── TASK-0040
TASK-0041
```

（TASK-0039、TASK-0040、TASK-0041 は並行実行可能。TASK-0039 と TASK-0040 はそれぞれ Phase 16 のタスクに依存）

---

## Phase 18: ドキュメント更新

**期間**: 1日
**目標**: Phase 16-17 の完了を反映し、overview.md、acceptance-criteria.md を最新状態に更新
**成果物**: 更新済みドキュメント

### タスク一覧

- [x] [TASK-0042: Phase 16-17 ドキュメント更新](TASK-0042.md) - 1h (DIRECT) 🔵 ✅

### 依存関係

```
TASK-0036 ~ TASK-0041 → TASK-0042
```

---

## Phase 19: Perplexity E2E・Property-Based Testing

**期間**: 2日
**目標**: AI_HUB_MAKE_RUN_FEEDBACK指摘に基づき、perplexityパイプラインのE2E統合テスト、trainer間パリティ検証、accept()のプロパティベーステストを追加する
**成果物**: E2E統合テスト、パラメータ化パリティテスト、hypothesisプロパティテスト

### タスク一覧

- [x] [TASK-0043: Perplexity E2Eパイプライン統合テスト](TASK-0043.md) - 2h (TDD) 🔵 ✅
- [x] [TASK-0044: Trainer間perplexity配管パリティテスト](TASK-0044.md) - 1.5h (TDD) 🔵 ✅
- [x] [TASK-0045: accept()プロパティベーステスト（hypothesis）](TASK-0045.md) - 2.5h (TDD) 🟡 ✅

### 依存関係

```
TASK-0043 ──┐
TASK-0044 ──┼── ドキュメント更新
TASK-0045 ──┘
```

（TASK-0043、TASK-0044、TASK-0045 は並行実行可能）

---

## Phase 21: DRYリファクタリングとコード品質

**期間**: 1日
**目標**: トレーニングパイプラインの重複コードを解消し、保守性を向上させる
**成果物**: 共有バッチイテレータ、戦略リスト統一、チェックポイントヘルパ

### タスク一覧

- [x] [TASK-0049: DRYリファクタリング（InfiniteBatchIterator・StrategyList・CheckpointHelper）](TASK-0049.md) - 2h (DIRECT) 🔵 ✅
- [x] [TASK-0050: Phase 21 全テストスイート検証とoverview更新](TASK-0050.md) - 1h (DIRECT) 🔵 ✅

### 依存関係

```
TASK-0049 → TASK-0050
```

**期間**: 2日
**目標**: Phase 19の完了を文書に反映し、GPUテストのOOM保護を追加してテストスイートの全通過を確認する
**成果物**: 完了済みドキュメント、堅牢化されたGPUテスト、全通過確認済みテストスイート

### タスク一覧

- [x] [TASK-0046: Phase 19 完了ドキュメント更新](TASK-0046.md) - 1h (DIRECT) 🔵 ✅
- [x] [TASK-0047: GPUテストOOM保護とリソース競合対策](TASK-0047.md) - 2h (TDD) 🔵 ✅
- [x] [TASK-0048: テストスイート全通過確認とリグレッションテスト](TASK-0048.md) - 1h (DIRECT) 🔵 ✅

### 依存関係

```
TASK-0046 ──┬── TASK-0048
TASK-0047 ──┘
```

（TASK-0046 と TASK-0047 は並行実行可能）

---

## Phase 22: 公開API・入力検証・エッジケース強化

**期間**: 1日
**目標**: AI_HUB_MAKE_RUN_FEEDBACK指摘に基づき、公開APIエクスポート、入力検証、評価エッジケース強化を実装する
**成果物**: 公開API、入力検証、エッジケース強化

### 成果（コミット内実装）

- REQ-073: src/tg_lora/__init__.py 公開APIエクスポート ✅
- REQ-074: RandomWalkController 入力検証 ✅
- REQ-075: eval_loss空データローダーNaN返却 ✅
- REQ-076: rollback try-catch安全性 ✅
- REQ-077: 非有限loss_afterガード ✅

### 関連コミット

- `cceccde feat(tg_lora): add public API exports, input validation, and robustness improvements`
- `e19da0f fix(training): harden eval loss edge cases, rollback safety, and non-finite loss guard`

---

## Phase 23: テストカバレッジ強化と堅牢性向上

**期間**: 1日
**目標**: Phase 21で抽出された共有ユーティリティのテストカバレッジを完成させ、運用堅牢性を向上させる
**成果物**: checkpoint readback検証、batch iterator エッジケーステスト、warning log、rollback E2Eテスト

### タスク一覧

- [x] [TASK-0051: save_checkpoint readback検証テスト](TASK-0051.md) - 2h (TDD) 🔵 ✅
- [x] [TASK-0052: InfiniteBatchIterator エッジケーステスト](TASK-0052.md) - 2h (TDD) 🔵 ✅
- [x] [TASK-0053: 非有限loss_after warning log追加](TASK-0053.md) - 1h (TDD) 🟡 ✅
- [x] [TASK-0054: RollbackManager rollback例外E2Eテスト](TASK-0054.md) - 2h (TDD) 🔵 ✅
- [x] [TASK-0055: Phase 23 全テストスイート検証とoverview更新](TASK-0055.md) - 1h (DIRECT) 🔵 ✅

### 依存関係

```
TASK-0051 ──┐
TASK-0052 ──┼── TASK-0055
TASK-0053 ──┤
TASK-0054 ──┘
```

（TASK-0051、TASK-0052、TASK-0053、TASK-0054 は並行実行可能）

---

## 次のステップ

Phase 25 が完了しました。Phase 24, 26 の残タスクを実装するには:

- 全タスク順番に実装: `/tsumiki:kairo-implement`
- 特定タスクを実装: `/tsumiki:kairo-implement TASK-0056`

---

## Phase 24: MLflow実験管理高度化

**期間**: 2日
**目標**: MLflow統合を完成させ、実験管理・チェックポイント保存・メトリクス追跡を高度化する
**成果物**: アーティファクトロギング、ランメタデータ自動生成、TG-LoRA特化メトリクス、リトライロジック

### タスク一覧

- [x] [TASK-0056: MLflowアーティファクトロギング統合](TASK-0056.md) - 3h (TDD) 🔵 ✅
- [x] [TASK-0057: MLflowランメタデータ自動生成](TASK-0057.md) - 2h (TDD) 🔵 ✅
- [x] [TASK-0058: TG-LoRA特化メトリクスMLflow統合](TASK-0058.md) - 2.5h (TDD) 🔵 ✅
- [x] [TASK-0059: MLflowリトライロジックとエラー強化](TASK-0059.md) - 2h (TDD) 🔵🟡 ✅

### 依存関係

```
TASK-0056 → TASK-0057 → TASK-0058 → TASK-0059
```

---

## Phase 25: 実験分析ツール整備

**期間**: 2日
**目標**: RunMetrics履歴のクエリAPI、複数ラン比較CLI、学習曲線可視化の強化
**成果物**: クエリAPI、ラン比較ダッシュボード、追加プロット関数

### タスク一覧

- [x] [TASK-0060: RunMetrics履歴クエリAPI](TASK-0060.md) - 3h (TDD) 🔵 ✅
- [x] [TASK-0061: ラン比較CLI・ダッシュボード](TASK-0061.md) - 3h (TDD) 🔵 ✅
- [x] [TASK-0062: 学習曲線可視化ユーティリティ強化](TASK-0062.md) - 2h (TDD) 🟡 ✅

### 依存関係

```
TASK-0059 → TASK-0060 → TASK-0061 → TASK-0062
```

---

## Phase 26: 本番運用品質

**期間**: 2日
**目標**: 障害回復機能、運用ランブック、全テスト検証による本番運用品質の確立
**成果物**: 自動リスタート、ランブック、APIリファレンス、全テスト確認済み

### タスク一覧

- [x] [TASK-0063: 学習ジョブ障害回復・自動リスタート](TASK-0063.md) - 3h (TDD) 🔵 ✅
- [x] [TASK-0064: 運用ランブック・APIリファレンス整備](TASK-0064.md) - 2h (DIRECT) 🔵 ✅
- [x] [TASK-0065: Phase 24-26 全テスト検証・ドキュメント更新](TASK-0065.md) - 1h (DIRECT) 🔵 ✅

### 依存関係

```
TASK-0062 → TASK-0063 → TASK-0064 → TASK-0065
```

---

## Phase 29: コード品質・整合性

**期間**: 1日
**目標**: ruff lint/formatの100エラー・52ファイルフォーマット不整合を修正し、overview.mdのテスト数整合性を確保する
**成果物**: クリーンアップ済みコード、整合性確認済みoverview

### タスク一覧

- [x] [TASK-0066: Ruff lint・format クリーンアップ](TASK-0066.md) - 2h (DIRECT) 🔵 ✅
- [x] [TASK-0067: テストスイート全通過確認とoverview整合性更新](TASK-0067.md) - 1.5h (DIRECT) 🔵 ✅

### 依存関係

```
TASK-0066 → TASK-0067
```

---

## Phase 30: OptimizerLifecycleManager E2E検証

**期間**: 1日
**目標**: AI_HUB_MAKE_RUN_FEEDBACK指摘に基づき、OptimizerLifecycleManagerのポリシー設定がrun_metrics出力に正しく現れることをE2E検証し、ベンチマークスクリプトのスモークテストを追加する
**成果物**: E2Eスモークテスト、ベンチマークテスト、更新済みドキュメント

### タスク一覧

- [x] [TASK-0068: OptimizerLifecycleManager E2E スモークテスト](TASK-0068.md) - 2h (TDD) 🔵 ✅
- [x] [TASK-0069: ベンチマークスクリプトスモークテスト](TASK-0069.md) - 1.5h (TDD) 🔵 ✅
- [x] [TASK-0070: Phase 30 ドキュメント更新](TASK-0070.md) - 1h (DIRECT) 🔵 ✅

### 依存関係

```
TASK-0067 ──┬── TASK-0068 ──┐
            └── TASK-0069 ──┼── TASK-0070
```

（TASK-0068 と TASK-0069 は並行実行可能）

---

## Phase 31: Makefile検証と新機能統合テスト

**期間**: 1日
**目標**: AI_HUB_MAKE_RUN_FEEDBACK指摘に基づき、Makefileターゲットの検証可能性を確立し、直近4コミットで追加された新機能（trainable_lora_scope, prefix_feature_cache）の統合テストを追加する
**成果物**: Makefileターゲット検証テスト、trainable_lora_scope統合テスト、prefix_feature_cache拡張テスト、実験config検証テスト、更新済みドキュメント

### タスク一覧

- [x] [TASK-0071: Makefile smoke・ablation・bench-optimizer ターゲット検証](TASK-0071.md) - 2.5h (TDD) 🔵 ✅
- [x] [TASK-0072: trainable_lora_scope 統合テスト](TASK-0072.md) - 2h (TDD) 🔵 ✅
- [x] [TASK-0073: prefix_feature_cache 拡張テスト](TASK-0073.md) - 2h (TDD) 🔵 ✅
- [x] [TASK-0074: Makefile 実験configターゲット配線検証](TASK-0074.md) - 1.5h (TDD) 🔵 ✅
- [x] [TASK-0075: Phase 31 ドキュメント更新](TASK-0075.md) - 1h (DIRECT) 🔵 ✅

### 依存関係

```
TASK-0070 ──┬── TASK-0071 ──── TASK-0074 ──┐
            ├── TASK-0072 ─────────────────┼── TASK-0075
            └── TASK-0073 ─────────────────┘
```

（TASK-0071, TASK-0072, TASK-0073 は並行実行可能。TASK-0074 は TASK-0071 に依存）

---

## Phase 33: AsyncCacheBuilder Acceptance・境界値テスト

**期間**: 1日
**目標**: REQ-136~138のacceptance criteriaを追加し、AsyncCacheBuilderの境界値テスト（force_rebuild、部分失敗、非存在label）を実装し、overview.mdのテスト数を実測値に修正する
**成果物**: REQ-136~138 acceptance criteria、AsyncCacheBuilder境界値テスト3件、修正済みoverview.md

### タスク一覧

- [x] [TASK-0076: REQ-136~138 acceptance criteria追加](TASK-0076.md) - 2h (TDD) 🔵 ✅
- [x] [TASK-0077: AsyncCacheBuilder境界値テスト追加](TASK-0077.md) - 1.5h (TDD) 🔵 ✅
- [x] [TASK-0078: Phase 33 overview.md更新とテスト数修正](TASK-0078.md) - 0.5h (DIRECT) 🔵 ✅

### 依存関係

```
TASK-0076 ── TASK-0077 ── TASK-0078
```

---

## Phase 34: パフォーマンス検証とテスト数同期

**期間**: 1日
**目標**: コミット851041eのin-place tensor ops最適化のdata_ptr保存検証テストを追加し、マイクロベンチマークで性能メリットを定量的に確認し、overview.mdのテスト数を実測値に同期する
**成果物**: data_ptr保存検証テスト、マイクロベンチマークスクリプト、更新済みoverview.md

### タスク一覧

- [x] [TASK-0079: In-place tensor ops data_ptr保存検証テスト](TASK-0079.md) - 2h (TDD) 🔵 ✅
- [x] [TASK-0080: Velocity EMA・cap_updateマイクロベンチマーク](TASK-0080.md) - 2h (TDD) 🔵 ✅
- [x] [TASK-0081: Phase 34 overview.md更新とテスト数同期](TASK-0081.md) - 1h (DIRECT) 🔵 ✅

### 依存関係

```
TASK-0079 ──┬── TASK-0081
TASK-0080 ──┘
```

（TASK-0079 と TASK-0080 は並行実行可能）

---

## Phase 36: スクリプト健全性とConfig検証完全性

**期間**: 1日
**目標**: 未テストスクリプトのインポート健全性テスト追加、スクリプトConfig YAML Schema検証統合
**成果物**: テスト追加、設定検証統合、更新済みoverview

### タスク一覧

- [x] [TASK-0083: scripts/inspect_model.py・summarize_sweep.py インポート健全性テスト](TASK-0083.md) - 1.5h (TDD) 🔵 ✅
- [x] [TASK-0084: スクリプトConfig YAML Schema検証統合](TASK-0084.md) - 2h (TDD) 🔵 ✅
- [x] [TASK-0085: Phase 35-36 overview.md更新とテスト数同期](TASK-0085.md) - 0.5h (DIRECT) 🔵 ✅

### 依存関係

```
TASK-0083 ──┬── TASK-0085
TASK-0084 ──┘
```

（TASK-0083 と TASK-0084 は並行実行可能）

---

## Phase 37: LR探索統合とpropose→training loop配線

**期間**: 0.5日
**目標**: log-normal LR explorationがtraining loopで正しく消費されることを確認
**成果物**: proposal.lrのstate反映、統合テスト3件、REQ-150~152

### タスク一覧

- [x] [TASK-0086: LR探索統合・propose→training loop配線](TASK-0086.md) - 1h (DIRECT) 🔵 ✅

---

## Phase 38: Config完全性とテストスイート品質

**期間**: 1日
**目標**: LR探索パラメータの全Config明示化、テストスイート69件警告解消、ドキュメント同期
**成果物**: 明示的Config YAML、警告解消済みテストスイート、更新済み受け入れ基準

### タスク一覧

- [x] [TASK-0087: LR探索Config明示化と全Config検証](TASK-0087.md) - 2h (TDD) 🔵 ✅
- [x] [TASK-0088: テストスイート警告解消と品質向上](TASK-0088.md) - 1.5h (TDD) 🔵 ✅
- [x] [TASK-0089: Phase 38 ドキュメント同期と受け入れ基準更新](TASK-0089.md) - 1h (DIRECT) 🔵 ✅

### 依存関係

```
TASK-0087 ──┬── TASK-0089
TASK-0088 ──┘
```

（TASK-0087 と TASK-0088 は並行実行可能）

---

## Phase 40: --resume E2E統合テスト

**期間**: 1日
**目標**: AI_HUB_MAKE_RUN_FEEDBACK指摘に基づき、--resumeのsave→interrupt→resume→loss継続検証のE2E統合テストを追加する
**成果物**: tests/test_resume_e2e.py（E2E resume flow, loss continuity, cycle skipping, velocity preservation）

### タスク一覧

- [x] [TASK-0090: --resume E2E統合テスト（save→interrupt→resume→verify loss）](TASK-0090.md) - 3h (TDD) 🔵 ✅

### 依存関係

```
TASK-0089 ── TASK-0090 ── TASK-0093
```

---

## Phase 41: TruthfulQA分析とaccel param実験

**期間**: 1日
**目標**: TruthfulQAベンチマークのdelta -0.00045を調査し、accel adaptation paramsのチューニングで品質ギャップを埋めるための分析・実験基盤を構築する
**成果物**: ベンチマーク分析スクリプト、accel param感度テスト、実験config群、スイープスクリプト

### タスク一覧

- [x] [TASK-0091: TruthfulQAベンチマーク結果分析とaccel param効果調査](TASK-0091.md) - 2h (TDD) 🔵🟡
- [x] [TASK-0092: Accel adaptation param実験config作成とスイープスクリプト](TASK-0092.md) - 2h (DIRECT) 🔵

### 依存関係

```
TASK-0090 ── TASK-0091 ── TASK-0092 ── TASK-0093
```

---

## Phase 42: ドキュメント更新

**期間**: 0.5日
**目標**: Phase 40-41の完了をoverview.md、acceptance-criteria.md、architecture.mdに反映する
**成果物**: 更新済みドキュメント3ファイル

### タスク一覧

- [x] [TASK-0093: Phase 40-41 ドキュメント更新](TASK-0093.md) - 1h (DIRECT) 🔵

### 依存関係

```
TASK-0090 ──┬── TASK-0093
TASK-0091 ──┤
TASK-0092 ──┘
```

---

## Phase 43: Sweep実行と結果分析

**期間**: 1日
**目標**: TASK-0092の4つのaccel param実験configをGPUで実行し、比較分析して最適パラメータを特定する。TruthfulQA品質ギャップ（delta -0.00045 acc）の改善を検証
**成果物**: 4学習run、pairwise比較レポート、dashboard overview、品質評価結果

### タスク一覧

- [ ] [TASK-0094: Accel param sweep実行と結果分析](TASK-0094.md) - 4h (DIRECT) 🔵🟡

### 依存関係

```
TASK-0092 ── TASK-0094
```

### 実験分離設計

全configで統一パラメータ（単一変動軸: accel_instability_lr_decay × accel_convergence_lr_boost）:
- `enable_random_walk: false` — 確定的比較
- 同一のK/N/alpha/beta/lr初期値・候補リスト・layer strategy・seed

| Config | decay | boost | 効果 |
|--------|-------|-------|------|
| no_accel | 0.99 | 1.01 | 基準線（near-identity） |
| conservative | 0.3 | 1.1 | 強い保守 + 控えめ回復 |
| balanced | 0.5 | 1.5 | バランス型 |
| aggressive | 0.9 | 2.0 | 弱い保守 + 強い回復 |

---

## Phase 44: コンストラクタ検証（学習インフラ）

**期間**: 1日
**目標**: 学習インフラクラス（OptimizerLifecycleManager, InfiniteBatchIterator, LoraDataset系）のコンストラクタにパラメータ検証を追加する。前イテレーションで確立した検証パターンを適用
**成果物**: 6クラスのコンストラクタ検証、対応する単体テスト

### タスク一覧

- [x] [TASK-0095: OptimizerLifecycleManager + InfiniteBatchIterator コンストラクタ検証](TASK-0095.md) - 2h (TDD) 🔵 ✅
- [x] [TASK-0096: LoraDataset + PrefixFeatureDataset + MappedPrefixFeatureDataset コンストラクタ検証](TASK-0096.md) - 2h (TDD) 🔵 ✅

### 依存関係

```
TASK-0093 ── TASK-0095 ── TASK-0096
```

---

## Phase 45: コンストラクタ検証（ユーティリティ・評価）

**期間**: 1日
**目標**: ユーティリティ・評価クラス（MLflowLogger, RunMetrics, EvalLossResult, AsyncCacheBuilder）のコンストラクタにパラメータ検証を追加する
**成果物**: 4クラスのコンストラクタ検証、対応する単体テスト

### タスク一覧

- [x] [TASK-0097: MLflowLogger + RunMetrics + EvalLossResult コンストラクタ検証](TASK-0097.md) - 2h (TDD) 🔵 ✅
- [x] [TASK-0098: AsyncCacheBuilder コンストラクタ検証](TASK-0098.md) - 2h (TDD) 🔵 ✅

### 依存関係

```
TASK-0096 ── TASK-0097 ── TASK-0098
```

---

## Phase 46: フレイキーテスト排除

**期間**: 1日
**目標**: test_fault_recovery.pyとtest_random_walk_controller.pyの非決定論的テストを修正し、CI信頼性を向上させる
**成果物**: lr_explore_prob非決定論性修正、統計アサーション堅牢化

### タスク一覧

- [x] [TASK-0099: RandomWalkController lr_explore_prob 非決定論性修正](TASK-0099.md) - 1.5h (TDD) 🔵 ✅
- [x] [TASK-0100: 統計アサーション堅牢化](TASK-0100.md) - 1.5h (TDD) 🔵 ✅

### 依存関係

```
TASK-0098 ── TASK-0099 ── TASK-0100
```

---

## Phase 47: ドキュメント更新・検証

**期間**: 0.5日
**目標**: Phase 44-46の完了をoverview.md、_doc_spine.ymlに反映し、テストスイート全通過を確認する
**成果物**: 更新済みoverview.md、_doc_spine.yml

### タスク一覧

- [x] [TASK-0101: Phase 44-46 テストスイート検証とoverview更新](TASK-0101.md) - 1h (DIRECT) 🔵 ✅

### 依存関係

```
TASK-0095 ~ TASK-0100 ── TASK-0101
```

---

## Phase 48: NaN/Inf バリデーション・ランタイムガード完全化

**期間**: 1日
**目標**: AI_HUB_MAKE_RUN_FEEDBACK指摘に基づき、accel param境界値テストでNaN/Inf入力ギャップを発見し、コンストラクタ検証・ランタイムガード・スクリプト防御の3層で修正する
**成果物**: NaN/Inf バリデーション修正（バグ修正）、ランタイムガード、11新規テスト

### タスク一覧

- [x] [TASK-0102: NaN/Inf バリデーションとランタイムガード完全化](TASK-0102.md) - 3h (TDD) 🔵 ✅

### 依存関係

```
TASK-0101 ── TASK-0102
```

### 発見バグ

- `accel_convergence_lr_boost` が NaN/Inf を受け入れる（`nan <= 1.0` → False でバイパス）
- pydantic `gt=1.0` が `inf` を受け入れる
- `adapt_to_acceleration()` が NaN/Inf acceleration で誤動作
- `analyze_accel_sweep.py` が NaN/None baseline loss で TypeError/NaN propagatio

---

## Phase 49: テスト数同期とCI gate安定性

**期間**: 1日
**目標**: Phase 48以降に追加されたテストによるoverview.mdテスト数乖離（1999→2032）を解消し、CI gate baselineテストの安定性を向上させる
**成果物**: 同期済みoverview.md、安定化されたCI gateテスト

### タスク一覧

- [x] [TASK-0103: Phase 49 テスト数同期・ドキュメント整合性更新](TASK-0103.md) - 1h (DIRECT) 🔵 ✅
- [x] [TASK-0104: CI gate baselineテスト安定性調査と修正](TASK-0104.md) - 2h (TDD) 🔵 ✅

### 依存関係

```
TASK-0102 ── TASK-0103 ── TASK-0104
```

---

## Phase 50: Stage 2 マルチシード複製・Paper Gate評価

**期間**: 2日
**目標**: Stage 2 multi-seed replicationを実行しGate G0/G1のpass/failを確定する。paper_experiment_plan.mdに基づくGate評価自動化スクリプトを実装する
**成果物**: evaluate_paper_gates.py, Stage 2実行手順, gate_report

### タスク一覧

- [x] [TASK-0105: Paper Gate評価自動化スクリプト](TASK-0105.md) - 2h (TDD) 🔵 ✅
- [ ] [TASK-0106: Stage 2 マルチシード実験実行とGate評価](TASK-0106.md) - 3h (実験) 🔵

### 依存関係

```
TASK-0105 ── TASK-0106
```

---

## Phase 51: Paper Pipeline Stage 3-5自動化

**期間**: 2日
**目標**: Stage 3 frontier sweep、Stage 4外部品質評価、因果分析（G4）の自動化スクリプトを実装し、Stage 2実行前のsmoke検証を強化する
**成果物**: run_frontier_sweep.sh, run_paper_external_eval.py, evaluate_paper_gates.py G4拡張, dry-run検証

### タスク一覧

- [x] [TASK-0107: Stage 3 メモリフロンティアスイープ自動化スクリプト](TASK-0107.md) - 4h (TDD) 🔵 ✅
- [ ] [TASK-0108: 外部品質評価パイプライン（G3 Gate）](TASK-0108.md) - 3h (TDD) 🔵
- [x] [TASK-0109: 因果分析評価ロジック拡張（G4 Gate）](TASK-0109.md) - 3h (TDD) 🔵 ✅
- [x] [TASK-0111: Stage 2 実行前Smoke検証強化](TASK-0111.md) - 2h (TDD) 🔵 ✅

### 依存関係

```
TASK-0106 ── TASK-0107 ── TASK-0110
TASK-0106 ── TASK-0108 ── TASK-0110
TASK-0105 ── TASK-0109 ── TASK-0110
TASK-0105 ── TASK-0111 ── TASK-0106
```

---

## Phase 52: 論文結果統合

**期間**: 1日
**目標**: Stage 2-5の全実験結果を統合し、論文に直接転記可能なテーブル・Claim Ladder判定を自動生成する
**成果物**: consolidate_paper_results.py, LaTeX/Markdownテーブル

### タスク一覧

- [x] [TASK-0110: 論文結果統合・テーブル自動生成スクリプト](TASK-0110.md) - 2h (DIRECT) 🔵 ✅

### 依存関係

```
TASK-0107 ── TASK-0110
TASK-0108 ── TASK-0110
TASK-0109 ── TASK-0110
```

---

## Phase 56: モデル検査・比較ダッシュボード・ワンショットキャッシュ・コスト分析

**期間**: 1日
**目標**: Phase 55完了後の要件ギャップを解消し、モデル検査ツール・比較ダッシュボード・ワンショットキャッシュモード・コスト分析スクリプトの要件を正式化する
**成果物**: 14件の新規要件（REQ-218~231）、対応するacceptance criteria・taskファイル

### タスク一覧

- [ ] [TASK-0112: モデル検査・比較ダッシュボード・ワンショットキャッシュのacceptance criteria追加](TASK-0112.md) - 2h (TDD) 🔵
- [ ] [TASK-0113: コスト分析・データ細粒度・クリーンアップターゲットのテスト追加](TASK-0113.md) - 1h (TDD) 🔵

### 依存関係

```
TASK-0112 → TASK-0113
```

---

## Phase 57: 論文実験統計分析強化

**期間**: 2日
**目標**: マルチシード実験結果の統計分析モジュール・論文エクスポートツール・ハイパーパラメータ感度分析を追加し、論文実験パイプラインの分析能力を強化する
**成果物**: src/analysis/stats.py, scripts/export_paper_results.py, scripts/analyze_sensitivity.py

### タスク一覧

- [x] [TASK-0114: マルチシード統計分析モジュール](TASK-0114.md) - 3h (TDD) 🟡 ✅
- [x] [TASK-0115: 論文結果エクスポートツール](TASK-0115.md) - 2h (TDD) 🟡 ✅
- [x] [TASK-0116: ハイパーパラメータ感度分析ツール](TASK-0116.md) - 2h (TDD) 🟡 ✅

### 依存関係

```
TASK-0112 → TASK-0114 → TASK-0115 → TASK-0116
```

---

## Phase 58: 学習品質モニタリング

**期間**: 1日
**目標**: 学習サイクル健全性モニターとクロス構成実験コンパレータを追加し、学習品質の自動監視と実験横断比較能力を強化する
**成果物**: src/tg_lora/cycle_monitor.py, scripts/compare_experiment_configs.py

### タスク一覧

- [x] [TASK-0117: 学習サイクル健全性モニター](TASK-0117.md) - 2h (TDD) 🟡 ✅
- [x] [TASK-0118: クロス構成実験マトリクスコンパレータ](TASK-0118.md) - 2h (TDD) 🟡 ✅

### 依存関係

```
TASK-0116 → TASK-0117 → TASK-0118
TASK-0114 ─────────────→ TASK-0118
```

---

## Phase 59: 学習軌跡分析・収束予測・早期停止推奨

**期間**: 1日
**目標**: 学習loss履歴から収束予測・早期停止推奨・異常検知を行うTrajectoryAnalyzerモジュールとCLIツールを実装し、学習の意思決定を支援する
**成果物**: src/tg_lora/trajectory.py, scripts/analyze_trajectory.py, tests/test_trajectory.py

### タスク一覧

- [x] [TASK-0119: 学習軌跡分析モジュール・CLI・テスト](TASK-0119.md) - 3h (TDD) 🟡 ✅

### 依存関係

```
TASK-0118 → TASK-0119
```

---

## Phase 60: 軌跡連動適応制御

**期間**: 1日
**目標**: TrajectoryAnalyzerの軌跡分析結果に基づいてRandomWalkControllerのパラメータをリアルタイムに適応させるTrajectoryControllerモジュールを実装し、学習の自動最適化を実現する
**成果物**: src/tg_lora/trajectory_controller.py, tests/test_trajectory_controller.py

### タスク一覧

- [x] [TASK-0120: 軌跡連動適応制御モジュール・テスト](TASK-0120.md) - 3h (TDD) 🟡 ✅

### 依存関係

```
TASK-0119 → TASK-0120
```

---

## Phase 61: Training Advisor モジュール・CLI

**期間**: 1日
**目標**: CycleMonitorとTrajectoryAnalyzerを統合し、優先順位付きアクションを生成するTraining AdvisorモジュールとCLIツールを実装し、学習の意思決定を包括的に支援する
**成果物**: src/tg_lora/training_advisor.py, scripts/advise_training.py, tests/test_training_advisor.py

### タスク一覧

- [x] [TASK-0121: Training Advisor モジュール・CLI・テスト](TASK-0121.md) - 3h (TDD) 🟡 ✅

### 依存関係

```
TASK-0117 → TASK-0121
TASK-0119 → TASK-0121
```

---

## Phase 62: 統合テスト・CI品質強化

**期間**: 1日
**目標**: AI Hub feedbackに基づき、新規CLIスクリプトのE2E統合テスト、CLIスモークテスト、CI回帰テストMakefileターゲットを追加し、自動検証品質を強化する
**成果物**: E2E統合テスト2ファイル、CLIスモークテスト、Makefileターゲット2件

### タスク一覧

- [x] [TASK-0122: advise_training.py E2E統合テスト](TASK-0122.md) - 3h (TDD) 🔵 ✅ 2026-05-25
- [x] [TASK-0123: analyze_trajectory.py E2E統合テスト](TASK-0123.md) - 2h (TDD) 🔵 ✅ 2026-05-25
- [x] [TASK-0124: CLIスモークテスト Makefile ターゲット](TASK-0124.md) - 1.5h (DIRECT) 🔵 ✅ 2026-05-25
- [x] [TASK-0125: CI回帰テスト Makefile ターゲット](TASK-0125.md) - 1.5h (DIRECT) 🔵 ✅ 2026-05-25

### 依存関係

```
TASK-0121 → TASK-0122 → TASK-0125
TASK-0119 → TASK-0123 → TASK-0125
TASK-0124 → TASK-0125
```

（TASK-0122, TASK-0123, TASK-0124 は並行実行可能）

---

## Phase 63: 最終仕様整合性と品質確認

**期間**: 0.5日
**目標**: requirements.mdの黄信号要件ステータスを実装コードと照合して更新し、run_eval_lora.shにtrap handlerを追加し、overview.mdの進捗情報を最新化する
**成果物**: 6件の🟡→🔵更新、trap handler追加、最新化されたoverview

### タスク一覧

- [x] [TASK-0126: requirements.md 黄信号要件ステータスの実態反映](TASK-0126.md) - 1h (DIRECT) 🔵 ✅ 2026-05-25
- [x] [TASK-0127: run_eval_lora.sh 終了時trap handler追加](TASK-0127.md) - 0.5h (DIRECT) 🔵 ✅ 2026-05-25
- [x] [TASK-0128: overview.md フェーズ進捗とテスト状況の最新化](TASK-0128.md) - 1h (DIRECT) 🔵 ✅ 2026-05-25

### 依存関係

```
TASK-0126 ── TASK-0128
TASK-0127
```

（TASK-0126 と TASK-0127 は並行実行可能）

---

## Phase 64: parse_warnings E2E検証

**期間**: 0.5日
**目標**: 意図的に破損したJSONLエントリでparse_warningsの全パイプライン（gather → dashboard → JSON出力）を検証する
**成果物**: 10件のE2Eテスト、requirements.md EDGE-003/136/REQ-071 🟡→🔵更新

### タスク一覧

- [x] [TASK-0129: parse_warnings corrupt-JSONL end-to-end integration tests](TASK-0129.md) - 1.5h (TDD) 🔵 ✅ 2026-05-25

### 依存関係

```
TASK-0126 ── TASK-0129
```

---

## Phase 65: Progressive Freezing E2E統合

**期間**: 0.5日
**目標**: Progressive Freezing機能群（freeze schedule + freeze frontier + activation-matching local loss）を実モデルの forward+backward で統合検証し、モジュール単体テストでは検出できない「部分凍結多層状態での勾配フロー・schedule/loss 相互作用」の回帰を捕捉する
**成果物**: 5件のE2E統合テスト（tests/test_progressive_freeze_e2e.py）

### タスク一覧

- [x] [TASK-0130: Progressive Freezing schedule+frontier+local-loss E2E integration test](TASK-0130.md) - 1.5h (TDD) 🔵 ✅ 2026-06-20

### 依存関係

```
activation_matching(Phase1) ──┐
freeze_schedule(Phase2) ──────┼── TASK-0130
freeze_frontier(Phase2) ──────┘
```

---

## Phase 66: Progressive Freezing 多層化（design §4.1）

**期間**: 0.5日
**目標**: `ProgressiveFreezeController` を単層ゲート（Phase 1）から、`FreezeSchedule` に駆動される真の多層プログレッシブ凍結（design §4.1: X→X-1→X-2 と凍結集合が単調増加）へ昇格させる。Phase 65 の E2E は層ごとに別コントローラを生成するワークアラウンドでこの単層制約を回避していたが、単一コントローラでスケジュール全体を駆動する機構が欠けていた——「Progressive」Freezing の名の由来である核心機能を閉じる。
**成果物**: `progressive_freeze.py` の progressive API（`apply_freeze_layer` / `layers_due_at` / `progress` / 層別 xin キャッシュ / `compute_local_loss(layer_idx=)`）、15件の progressive ユニットテスト + 1件の単一コントローラ E2E（tests/test_progressive_freeze_progressive.py, test_progressive_freeze_e2e.py）。単層挙動は完全保持（既存33テスト緑）。

### タスク一覧

- [x] [TASK-0131: ProgressiveFreezeController 多層プログレッシブ凍結（design §4.1）](TASK-0131.md) - 2h (TDD) 🔵 ✅ 2026-06-20

### 依存関係

```
freeze_schedule(Phase2) ──┐
TASK-0130(Phase65 E2E) ───┴── TASK-0131
```

---

## 次のステップ

Phase 66 完了。残存する未完了タスク:

- **TASK-0094** (Phase 43): GPU依存のaccel param sweep実行
- **TASK-0106** (Phase 50): GPU依存のマルチシード実験実行
- **TASK-0108** (Phase 51): GPU依存の外部品質評価パイプライン

プロジェクトの実装・テスト・ドキュメントは実質的に完了状態（2538テスト全パス、カバレッジ99%、517/517受け入れ基準）。


<!-- spine:references:begin -->
## Spine: external references

- [TASK-0020: docs/llm-wiki git tracking 解除とクリーンアップ](TASK-0020.md)
- [TASK-0021: _InfiniteBatchIterator StopIteration テスト強化](TASK-0021.md)
- [TASK-0022: MLflow 実験ロギング統合](TASK-0022.md)
- [TASK-0023: GitHub Actions CI/CD パイプライン構築](TASK-0023.md)
- [TASK-0024: Docker 開発環境構築](TASK-0024.md)
- [TASK-0025: データスキーマバリデーション追加](TASK-0025.md)
- [TASK-0026: 比較レポート可視化強化](TASK-0026.md)
- [TASK-0027: GPU学習環境準備とモデル読み込み検証](TASK-0027.md)
- [TASK-0028: TG-LoRA 10サイクル学習スモークテスト](TASK-0028.md)
- [TASK-0029: ベースラインQLoRA学習実行](TASK-0029.md)
- [TASK-0030: 公正比較実験と結果分析](TASK-0030.md)
- [TASK-0031: Phase 9 受け入れ基準・ドキュメント更新](TASK-0031.md)
- [TASK-0032: 実外挿コードによるNaN検出統合テスト](TASK-0032.md)
- [TASK-0033: 多様化障害サイクル統合テスト](TASK-0033.md)
- [TASK-0034: Phase 14 信頼性修正の全テストスイート検証とリグレッション確認](TASK-0034.md)
- [TASK-0035: Phase 15 ドキュメント・受け入れ基準更新](TASK-0035.md)
- [TASK-0036: ベースライン学習にEvalLossResult統合](TASK-0036.md)
- [TASK-0037: 未使用Config項目の整理と早期停止パラメータ露出](TASK-0037.md)
- [TASK-0038: MLflow ロギングのベースライン/TG-LoRA間一貫性確保](TASK-0038.md)
- [TASK-0039: Layer Sampler temperatureパラメータ統合テスト](TASK-0039.md)
- [TASK-0040: RunMetrics perplexity出力とEvalLossResult E2Eテスト](TASK-0040.md)
- [TASK-0041: 空テストスタブ補完とエッジケースカバレッジ向上](TASK-0041.md)
- [TASK-0042: Phase 16-17 ドキュメント更新](TASK-0042.md)
- [TASK-0046: Phase 19 完了ドキュメント更新](TASK-0046.md)
- [TASK-0047: GPUテストOOM保護とリソース競合対策](TASK-0047.md)
- [TASK-0048: テストスイート全通過確認とリグレッションテスト](TASK-0048.md)
- [TASK-0050: Phase 21 全テストスイート検証とoverview更新](TASK-0050.md)
- [TASK-0066: Ruff lint・format クリーンアップ](TASK-0066.md)
- [TASK-0067: テストスイート全通過確認とoverview整合性更新](TASK-0067.md)
- [TASK-0078: Phase 33 overview.md更新とテスト数修正](TASK-0078.md)
- [TASK-0106: Stage 2 マルチシード実験実行とGate評価](TASK-0106.md)
- [TASK-0107: Stage 3 メモリフロンティアスイープ自動化スクリプト](TASK-0107.md)
- [TASK-0108: 外部品質評価パイプライン（G3 Gate）](TASK-0108.md)
- [TASK-0109: 因果分析評価ロジック拡張（G4 Gate）](TASK-0109.md)
- [TASK-0110: 論文結果統合・テーブル自動生成スクリプト](TASK-0110.md)
- [TASK-0111: Stage 2 実行前Smoke検証強化](TASK-0111.md)
- [TASK-0130: Progressive Freezing schedule+frontier+local-loss E2E integration test](TASK-0130.md)
- [TASK-0131: ProgressiveFreezeController 多層プログレッシブ凍結（design §4.1）](TASK-0131.md)
- [TASK-0132: §6.2 Level-1 証拠配管 — 計測された in-vivo 実現削減が ceiling を引き上げて回復する着地点（design §6.2）](TASK-0132.md)
- [TASK-0133: §6.3 比較ヘッドライン再生産 bracket 着地点 — A/B 再生産観測が additional_realized_reduction の bracket を較正する着地点（design §6.3 / Phase 3）](TASK-0133.md)
- [TASK-0134: §6.3 分散較正バンドの非負床注記 — `format_reduction_band` が負の下限を達成可能と読ませない着地点（design §6.3）](TASK-0134.md)
- [TASK-0135: §6.2/§6.3 証拠配管を CPU-proxy in-vivo A/B 実測へ配線 — 薄い証拠の着地点を本物の観測で厚くし、判定経路を観察可能にする（design §6.2/§6.3 → Phase 3 A/B）](TASK-0135.md)
- [TASK-0136: 起点発射性の repo-wide schema gate — 任意の LoggingConfig/スキーマ変更が出荷 config を壊すのを、feature コミットと同一コミットで捕える（d2218ed 回帰クラスの一般化）](TASK-0136.md)
- [TASK-0137: M10.3 disk-death guard の第2ベクトル閉包 — `trajectory_delta_artifacts/*.pt` のサイクル毎無限蓄積を、既存の keep_last_checkpoints/min_free_disk_gb knob で抑える](TASK-0137.md)
- [TASK-0138: M10.3 disk-death guard の第3ベクトル閉包 — ベースライン学習エントリポイント（`checkpoint-<step>` + `trajectory_delta_artifacts/*.pt`）を既存 knob 傘下へ](TASK-0138.md)
- [TASK-0139: Guard 実験 §4 可逆解放の不活性化欠陥閉包 — `DynamicFreezeController` の攪拌/上流活性リリースが同サイクルで即時再凍結される（リリースが no-op）+ §3/§4 制御カバレッジ](TASK-0139.md)
- [TASK-0140: Guard 実験 §5.2 品質ゲートの品質信号誤参照閉包 — 決定関数が pilot/split 境界 proxy 損失（`loss_valid`）を設計が要求する full-eval 損失（`valid_full`）の代わりに読んでいた + §5.1/§5.2/§5.3 決定関数の直接カバレッジ 0](TASK-0140.md)
- [TASK-0141: Guard §5.2 誠実性契約の最終一哩 — trainer が `loss_valid_full` を step record に永続化して受口を活性化させる](TASK-0141.md)
- [TASK-0142: 論文 Claim Gate G1/G4 の既知制限を正式記録 — gate 評価器に known-limitation を発火させ、具体 next action + owner で追跡可能にする](TASK-0142.md)
- [TASK-0143: §5.2 誠実性契約の休眠を可視化 — analyzer が pilot proxy に黙示フォールバックしたとき DORMANT 警告を発し TASK-0141 への所有を付与する](TASK-0143.md)
- [TASK-0144: 論文 Gate 評価器に INSUFFICIENT EVIDENCE 第3状態を導入 — 入力欠落 bail を FAIL（反証）と区別し、generic owner の G1/G4 task 誤帰属を修正](TASK-0144.md)
- [TASK-0149: G2 frontier-separation gate の INSUFFICIENT-EVIDENCE 契約閉包 — `_check_g2` が frontier report 欠損/破損時に disproven FAIL になっていた（G3/G4 は既に履行中・TASK-0144/a418049 の G2 残存 outlier）](TASK-0149.md)
- [TASK-0150: §5.2 誠実性契約の guard(B) 側閉包 — A/B 比較の休眠可視化が baseline(A) 側のみで、dormant な guard run が time-to-quality 比較を黙示汚染していた](TASK-0150.md)
- [TASK-0152: 9B target-scale 決定検証 recipe — Tier-1 turnkey 計測（9b candidate vs baseline・GOAL 究極目標）+ §4 verdict gate 拡張 gap（heterogeneous×generalize の surrogate arm 移植）。blocker は fundamental でなく runnable（upstream src.data + cached Qwen3.5-9B + RTX 3060 全て present・本 iter 検証）](TASK-0152.md)
- [TASK-0153: torn-write 整合性軸 LOAD 側閉包 — `load_trajectory_delta_artifact` に `CheckpointIntegrityError` 診断追加（8+ offline 解析 script 経由の Tier-2 dataset 不透明 crash → actionable fail-loud）+ integrity primitives の zero-dep leaf 化（`checkpoint_integrity.py`・`ed26173` の `atomic_save.py` と対・layering 保全）。feedback 指定の optimizer.pt（phantom・baseline は原子化+診断済 `training_state.pt` 内・TG-LoRA は永続化せず）/ TrainingState JSON（`.pt`-not-JSON）は検証で対象外確定・fabricate せず](TASK-0153.md)

<!-- spine:references:end -->
