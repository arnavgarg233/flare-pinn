@echo off
REM Quick CPU Smoke Test - Windows version
REM Takes ~30-45 minutes total on CPU

echo ========================================
echo SMOKE TEST: Baseline vs Physics-Informed
echo ========================================
echo.

REM Test 1: Data-only baseline (no physics)
echo Step 1/2: Training BASELINE (data-only, no physics)...
python src/train_pinn_improved.py --config configs/train_baseline_data_only.yaml --log logs/smoke_test_baseline.log

echo.
echo Baseline complete!
echo.
echo Step 2/2: Training PHYSICS-INFORMED model...

REM Test 2: Physics-informed
python src/train_pinn_improved.py --config configs/train_pinn_cpu_smoke_test.yaml --log logs/smoke_test_physics.log

echo.
echo ========================================
echo SMOKE TEST COMPLETE
echo ========================================
echo.
echo Compare results:
echo   Baseline:  checkpoints/baseline_data_only/best_model.pt
echo   Physics:   checkpoints/cpu_smoke_test/best_model.pt
echo.
echo Check logs:
echo   Baseline:  logs/smoke_test_baseline.log
echo   Physics:   logs/smoke_test_physics.log
echo.
echo Look for TSS@12h in the logs - physics should be higher!
pause

