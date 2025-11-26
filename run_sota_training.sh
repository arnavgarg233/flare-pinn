#!/bin/bash
# =============================================================================
# SOTA Training Pipeline for Solar Flare Prediction
# =============================================================================
# This script runs the 3-phase training pipeline:
#   Phase 1: Baseline (no physics) - Get TSS > 0.50
#   Phase 2: Physics-enhanced - Get TSS > 0.70
#   Phase 3: Ultimate - Push for TSS > 0.85
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  SOTA Training Pipeline for Flare-PINN${NC}"
echo -e "${BLUE}========================================${NC}"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python3 not found. Please install Python 3.8+${NC}"
    exit 1
fi

# =============================================================================
# PHASE 1: Baseline Training (No Physics)
# =============================================================================
echo -e "\n${YELLOW}PHASE 1: Baseline Training (Classification Only)${NC}"
echo -e "Goal: Achieve TSS > 0.50 at 24h horizon"
echo -e "Config: src/configs/baseline_sota.yaml"
echo ""

read -p "Run Phase 1 baseline training? (y/n): " run_phase1
if [[ $run_phase1 == "y" || $run_phase1 == "Y" ]]; then
    echo -e "${GREEN}Starting baseline training...${NC}"
    python3 -m src.train --config src/configs/baseline_sota.yaml
    
    echo -e "\n${GREEN}Phase 1 complete!${NC}"
    echo "Check outputs/logs/baseline_sota.log for results"
    echo "Look for: [Val] Max TSS > 0.50"
else
    echo -e "${YELLOW}Skipping Phase 1${NC}"
fi

# =============================================================================
# PHASE 2: Physics-Enhanced Training
# =============================================================================
echo -e "\n${YELLOW}PHASE 2: Physics-Enhanced Training${NC}"
echo -e "Goal: Achieve TSS > 0.70 at 24h horizon"
echo -e "Config: src/configs/sota_v2.yaml"
echo ""

read -p "Run Phase 2 physics training? (y/n): " run_phase2
if [[ $run_phase2 == "y" || $run_phase2 == "Y" ]]; then
    echo -e "${GREEN}Starting physics-enhanced training...${NC}"
    python3 -m src.train --config src/configs/sota_v2.yaml
    
    echo -e "\n${GREEN}Phase 2 complete!${NC}"
    echo "Check outputs/logs/sota_v2.log for results"
    echo "Look for: [Val] Max TSS > 0.70"
else
    echo -e "${YELLOW}Skipping Phase 2${NC}"
fi

# =============================================================================
# PHASE 3: Ultimate Training
# =============================================================================
echo -e "\n${YELLOW}PHASE 3: Ultimate Training (Maximum Performance)${NC}"
echo -e "Goal: Achieve TSS > 0.85 at 24h horizon"
echo -e "Config: src/configs/sota_ultimate.yaml"
echo ""

read -p "Run Phase 3 ultimate training? (y/n): " run_phase3
if [[ $run_phase3 == "y" || $run_phase3 == "Y" ]]; then
    echo -e "${GREEN}Starting ultimate training...${NC}"
    python3 -m src.train --config src/configs/sota_ultimate.yaml
    
    echo -e "\n${GREEN}Phase 3 complete!${NC}"
    echo "Check outputs/logs/sota_ultimate.log for results"
    echo "Look for: [Val] Max TSS > 0.85"
else
    echo -e "${YELLOW}Skipping Phase 3${NC}"
fi

# =============================================================================
# Summary
# =============================================================================
echo -e "\n${BLUE}========================================${NC}"
echo -e "${GREEN}Training Pipeline Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Checkpoints saved to:"
echo "  - outputs/checkpoints/baseline_sota/"
echo "  - outputs/checkpoints/sota_v2/"
echo "  - outputs/checkpoints/sota_ultimate/"
echo ""
echo "Best models saved as: best_model.pt in each directory"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Review validation metrics in log files"
echo "2. Run evaluation on test set"
echo "3. Apply temperature scaling for calibration"

