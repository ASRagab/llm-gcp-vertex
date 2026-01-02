#!/usr/bin/env bash
#
# Live integration tests for llm-gcp-vertex plugin
#
# Usage:
#   ./test_live.sh <project-id> [location]
#
# Examples:
#   ./test_live.sh my-gcp-project
#   ./test_live.sh my-gcp-project us-east1
#
# Prerequisites:
#   - gcloud auth application-default login
#   - llm installed and on PATH
#   - Claude models enabled in Model Garden (for Claude tests)
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
PASSED=0
FAILED=0
SKIPPED=0

# Parse arguments
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <project-id> [location]"
    echo ""
    echo "Arguments:"
    echo "  project-id  GCP project ID (required)"
    echo "  location    GCP region (default: us-central1)"
    exit 1
fi

PROJECT_ID="$1"
LOCATION="${2:-us-central1}"

# Export for llm plugin
export LLM_VERTEX_CLOUD_PROJECT="$PROJECT_ID"
export LLM_VERTEX_CLOUD_LOCATION="$LOCATION"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}llm-gcp-vertex Live Integration Tests${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "Project:  ${GREEN}$PROJECT_ID${NC}"
echo -e "Location: ${GREEN}$LOCATION${NC}"
echo ""

# Helper functions
log_test() {
    echo -e "${BLUE}[TEST]${NC} $1"
}

log_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((PASSED++)) || true
}

log_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((FAILED++)) || true
}

log_skip() {
    echo -e "${YELLOW}[SKIP]${NC} $1"
    ((SKIPPED++)) || true
}

# Install plugin from current directory
install_plugin() {
    echo -e "${BLUE}--- Setup ---${NC}"
    
    # Check if llm is available
    if ! command -v llm &> /dev/null; then
        echo -e "${RED}Error: llm command not found. Install with: pip install llm${NC}"
        exit 1
    fi
    
    # Always install from current directory in editable mode
    log_test "Installing plugin from current directory..."
    local install_output
    if install_output=$(llm install -e . 2>&1); then
        echo "$install_output" | tail -3
        # Verify models are now available (capture to avoid pipefail issues)
        local models_output
        models_output=$(llm models 2>&1) || true
        if echo "$models_output" | grep -q "gemini-2.5"; then
            log_pass "Plugin installed and models registered"
        else
            log_fail "Plugin installed but models not found"
            echo "Models output: $models_output" | head -5
            exit 1
        fi
    else
        echo "$install_output"
        log_fail "Could not install plugin"
        exit 1
    fi
    echo ""
}

# Test Gemini models
test_gemini() {
    echo -e "${BLUE}--- Gemini Tests ---${NC}"
    
    # Basic prompt (streaming by default)
    log_test "gemini-2.5-flash: basic prompt (streaming)..."
    if output=$(llm -m gemini-2.5-flash "Say 'hello' and nothing else" 2>&1); then
        if [[ "${output,,}" == *"hello"* ]]; then
            log_pass "Basic streaming prompt works"
        else
            log_fail "Unexpected response: $output"
        fi
    else
        log_fail "Command failed: $output"
    fi
    
    # Non-streaming
    log_test "gemini-2.5-flash: non-streaming (--no-stream)..."
    if output=$(llm -m gemini-2.5-flash --no-stream "Say 'test' and nothing else" 2>&1); then
        if [[ "${output,,}" == *"test"* ]]; then
            log_pass "Non-streaming works"
        else
            log_fail "Unexpected response: $output"
        fi
    else
        log_fail "Command failed: $output"
    fi
    
    # With temperature option
    log_test "gemini-2.5-flash: temperature option..."
    if output=$(llm -m gemini-2.5-flash -o temperature 0.1 "What is 2+2? Reply with just the number." 2>&1); then
        if [[ "$output" == *"4"* ]]; then
            log_pass "Temperature option works"
        else
            log_fail "Unexpected response: $output"
        fi
    else
        log_fail "Command failed: $output"
    fi
    
    # With max_output_tokens option
    log_test "gemini-2.5-flash: max_output_tokens option..."
    if output=$(llm -m gemini-2.5-flash -o max_output_tokens 10 "Count from 1 to 100" 2>&1); then
        # Should be truncated due to token limit
        log_pass "max_output_tokens option works (got ${#output} chars)"
    else
        log_fail "Command failed: $output"
    fi
    
    # With system prompt
    log_test "gemini-2.5-flash: system prompt..."
    if output=$(llm -m gemini-2.5-flash -s "You are a pirate. Always say 'Arrr' at the start." "Greet me" 2>&1); then
        if [[ "${output,,}" == *"arr"* ]]; then
            log_pass "System prompt works"
        else
            log_fail "Expected pirate response, got: $output"
        fi
    else
        log_fail "Command failed: $output"
    fi
    
    # Test gemini-2.5-pro (may be slower/more expensive)
    log_test "gemini-2.5-pro: basic prompt..."
    if output=$(llm -m gemini-2.5-pro "Say 'pro' and nothing else" 2>&1); then
        if [[ "${output,,}" == *"pro"* ]]; then
            log_pass "gemini-2.5-pro works"
        else
            log_fail "Unexpected response: $output"
        fi
    else
        log_fail "Command failed: $output"
    fi
    
    # Conversation test
    log_test "gemini-2.5-flash: conversation (-c)..."
    if llm -m gemini-2.5-flash "My name is TestUser" >/dev/null 2>&1; then
        if output=$(llm -m gemini-2.5-flash -c "What is my name?" 2>&1); then
            if [[ "${output,,}" == *"testuser"* ]]; then
                log_pass "Conversation works"
            else
                log_fail "Expected name recall, got: $output"
            fi
        else
            log_fail "Conversation continue failed: $output"
        fi
    else
        log_fail "Initial conversation message failed"
    fi
    
    echo ""
}

# Test Claude models
test_claude() {
    echo -e "${BLUE}--- Claude Tests ---${NC}"
    
    # Basic prompt (streaming by default)
    log_test "claude-sonnet-4.5: basic prompt (streaming)..."
    if output=$(llm -m claude-sonnet-4.5 "Say 'hello' and nothing else" 2>&1); then
        if [[ "${output,,}" == *"hello"* ]]; then
            log_pass "Basic streaming prompt works"
        else
            log_fail "Unexpected response: $output"
        fi
    else
        # Claude might not be enabled in Model Garden
        if [[ "$output" == *"not found"* ]] || [[ "$output" == *"permission"* ]] || [[ "$output" == *"not authorized"* ]]; then
            log_skip "Claude not enabled in Model Garden"
            log_skip "Skipping remaining Claude tests"
            echo ""
            return
        fi
        log_fail "Command failed: $output"
    fi
    
    # Non-streaming
    log_test "claude-sonnet-4.5: non-streaming (--no-stream)..."
    if output=$(llm -m claude-sonnet-4.5 --no-stream "Say 'test' and nothing else" 2>&1); then
        if [[ "${output,,}" == *"test"* ]]; then
            log_pass "Non-streaming works"
        else
            log_fail "Unexpected response: $output"
        fi
    else
        log_fail "Command failed: $output"
    fi
    
    # With temperature option
    log_test "claude-sonnet-4.5: temperature option..."
    if output=$(llm -m claude-sonnet-4.5 -o temperature 0.1 "What is 2+2? Reply with just the number." 2>&1); then
        if [[ "$output" == *"4"* ]]; then
            log_pass "Temperature option works"
        else
            log_fail "Unexpected response: $output"
        fi
    else
        log_fail "Command failed: $output"
    fi
    
    # With max_tokens option (Claude uses max_tokens, not max_output_tokens)
    log_test "claude-sonnet-4.5: max_tokens option..."
    if output=$(llm -m claude-sonnet-4.5 -o max_tokens 10 "Count from 1 to 100" 2>&1); then
        log_pass "max_tokens option works (got ${#output} chars)"
    else
        log_fail "Command failed: $output"
    fi
    
    # With system prompt
    log_test "claude-sonnet-4.5: system prompt..."
    if output=$(llm -m claude-sonnet-4.5 -s "You are a pirate. Always say 'Arrr' at the start." "Greet me" 2>&1); then
        if [[ "${output,,}" == *"arr"* ]]; then
            log_pass "System prompt works"
        else
            log_fail "Expected pirate response, got: $output"
        fi
    else
        log_fail "Command failed: $output"
    fi
    
    # With stop_sequences option
    log_test "claude-sonnet-4.5: stop_sequences option..."
    if output=$(llm -m claude-sonnet-4.5 -o stop_sequences '["STOP"]' "Say 'hello' then 'STOP' then 'world'" 2>&1); then
        if [[ "$output" != *"world"* ]]; then
            log_pass "stop_sequences option works"
        else
            log_fail "Stop sequence not honored: $output"
        fi
    else
        log_fail "Command failed: $output"
    fi
    
    # Conversation test
    log_test "claude-sonnet-4.5: conversation (-c)..."
    if llm -m claude-sonnet-4.5 "My name is ClaudeTestUser" >/dev/null 2>&1; then
        if output=$(llm -m claude-sonnet-4.5 -c "What is my name?" 2>&1); then
            if [[ "${output,,}" == *"claudetestuser"* ]]; then
                log_pass "Conversation works"
            else
                log_fail "Expected name recall, got: $output"
            fi
        else
            log_fail "Conversation continue failed: $output"
        fi
    else
        log_fail "Initial conversation message failed"
    fi
    
    echo ""
}

# Test multiple options combined
test_combined_options() {
    echo -e "${BLUE}--- Combined Options Tests ---${NC}"
    
    log_test "gemini-2.5-flash: multiple options combined..."
    if output=$(llm -m gemini-2.5-flash \
        -o temperature 0.5 \
        -o max_output_tokens 50 \
        -o top_p 0.9 \
        -o top_k 40 \
        -s "Be very brief." \
        "What is the capital of France?" 2>&1); then
        if [[ "${output,,}" == *"paris"* ]]; then
            log_pass "Multiple Gemini options work together"
        else
            log_fail "Unexpected response: $output"
        fi
    else
        log_fail "Command failed: $output"
    fi
    
    echo ""
}

# Test error handling
test_error_handling() {
    echo -e "${BLUE}--- Error Handling Tests ---${NC}"
    
    log_test "Invalid temperature (should fail gracefully)..."
    if output=$(llm -m gemini-2.5-flash -o temperature 5.0 "test" 2>&1); then
        log_fail "Should have failed with invalid temperature"
    else
        log_pass "Invalid temperature rejected"
    fi
    
    log_test "Invalid model name..."
    if output=$(llm -m nonexistent-model "test" 2>&1); then
        log_fail "Should have failed with invalid model"
    else
        log_pass "Invalid model rejected"
    fi
    
    echo ""
}

# Uninstall plugin
uninstall_plugin() {
    echo -e "${BLUE}--- Cleanup ---${NC}"
    log_test "Uninstalling plugin..."
    if llm uninstall llm-gcp-vertex -y >/dev/null 2>&1; then
        log_pass "Plugin uninstalled"
    else
        log_fail "Could not uninstall plugin"
    fi
    echo ""
}

# Print summary
print_summary() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Test Summary${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo -e "${GREEN}Passed:  $PASSED${NC}"
    echo -e "${RED}Failed:  $FAILED${NC}"
    echo -e "${YELLOW}Skipped: $SKIPPED${NC}"
    echo ""
    
    if [[ $FAILED -eq 0 ]]; then
        echo -e "${GREEN}All tests passed!${NC}"
    else
        echo -e "${RED}Some tests failed.${NC}"
    fi
}

# Main
main() {
    install_plugin
    
    # Use trap to ensure cleanup runs regardless of test outcome
    trap uninstall_plugin EXIT
    
    test_gemini
    test_claude
    test_combined_options
    test_error_handling
    print_summary
    
    # Return appropriate exit code
    [[ $FAILED -eq 0 ]]
}

main
