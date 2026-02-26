# Test Runner Documentation

## Overview

The Test Runner page (`/tests`) provides a comprehensive UI for executing all API test cases from `devforge_test_case.py`. It allows you to run tests individually or all at once, with detailed results and error reporting.

## Features

- ✅ **Run All Tests**: Execute all 12 test cases sequentially
- ✅ **Run Individual Tests**: Execute specific test cases
- ✅ **Real-time Status**: See test status (pending, running, passed, failed)
- ✅ **Detailed Results**: View response data, error messages, and request payloads
- ✅ **Edge ID Tracking**: Automatically tracks created edge IDs for dependent tests
- ✅ **Test Summary**: Overview of passed/failed/running tests

## Test Cases

The test runner includes all 12 test cases from `devforge_test_case.py`:

### 1. Create Node (doc1)
- Creates a node with ID "doc1"
- Text: "Redis caching strategies"
- Metadata: `{ type: "article", tags: ["cache", "redis"] }`
- Regenerates embedding

### 2. Get Node (doc1)
- Retrieves the node created in test 1
- Verifies node data

### 3. Update Node (doc1)
- Updates the node text and metadata
- Regenerates embedding

### 4. Create Node (doc4)
- Creates a second node for edge testing
- Text: "Database optimization techniques"

### 5. Create Edge
- Creates an edge between doc1 and doc4
- Type: "related_to"
- Weight: 0.8
- **Note**: The edge_id is automatically stored for tests 6, 7, and 11

### 6. Get Edge
- Retrieves the edge created in test 5
- Uses the stored edge_id automatically

### 7. Update Edge
- Updates the edge weight to 0.95
- Uses the stored edge_id automatically

### 8. Vector Search
- Searches for "redis caching"
- Top K: 5
- Metadata filter: `{ type: "guide" }`

### 9. Graph Traversal
- Traverses graph from doc1
- Depth: 2

### 10. Hybrid Search
- Searches for "redis caching"
- Vector weight: 0.6
- Graph weight: 0.4
- Top K: 5

### 11. Delete Edge
- Deletes the edge created in test 5
- Uses the stored edge_id automatically

### 12. Delete Node (doc7)
- Attempts to delete node doc7 (may not exist)

## Usage

### Running All Tests

1. Navigate to **Test Runner** page (`/tests`)
2. Click **Run All Tests** button
3. Watch tests execute sequentially
4. View results in real-time

### Running Individual Tests

1. Navigate to **Test Runner** page
2. Find the test case you want to run
3. Click the **Run** button on that test
4. View the result below the test

### Viewing Results

Each test shows:
- **Status**: Passed (green) or Failed (red)
- **Message**: Success or error message
- **Response**: Full API response data (expandable)
- **Error**: Detailed error information (if failed)

### Test Summary

The summary card shows:
- Total number of tests
- Number of passed tests
- Number of failed tests
- Number of running tests (during execution)
- Created Edge ID (if edge was created)

## Edge ID Handling

Tests 6, 7, and 11 depend on the edge_id created in test 5. The test runner:
- Automatically stores the edge_id when test 5 completes
- Uses the stored edge_id for dependent tests
- Displays the edge_id in the summary card
- Shows the actual edge_id in endpoint displays

## Error Handling

If a test fails:
- The test card turns red
- Error message is displayed
- Full error response is shown (expandable)
- Other tests continue to run

Common errors:
- **404 Not Found**: Node or edge doesn't exist
- **400 Bad Request**: Invalid payload or missing required fields
- **Network Error**: Backend not running or CORS issue

## Prerequisites

Before running tests:
1. ✅ Backend API must be running on `http://localhost:8000`
2. ✅ CORS must be enabled in backend (already configured)
3. ✅ Database should be clean (or tests may fail due to existing data)

## Test Execution Order

Tests are designed to run in order:
1. Create nodes (tests 1, 4)
2. Create edge (test 5)
3. Read operations (tests 2, 6)
4. Update operations (tests 3, 7)
5. Search operations (tests 8, 9, 10)
6. Delete operations (tests 11, 12)

**Note**: Running tests out of order may cause failures if dependencies aren't met.

## Resetting Tests

Click the **Reset** button to:
- Clear all test results
- Clear stored edge_id
- Prepare for a fresh test run

## Integration with Backend

The test runner uses the exact same API endpoints and payloads as `devforge_test_case.py`:
- Same request formats
- Same expected responses
- Same test logic
- Same error handling

This ensures consistency between UI testing and backend testing.

## Troubleshooting

### Tests Fail with 404
- Ensure backend is running
- Check that previous tests completed successfully
- Verify node/edge IDs exist

### Tests Fail with 400
- Check request payload format
- Verify required fields are present
- Check metadata format (must be valid JSON)

### Network Errors
- Verify backend URL is correct
- Check CORS configuration
- Ensure backend is accessible

### Edge ID Not Available
- Run test 5 (Create Edge) first
- Check that test 5 completed successfully
- Verify edge_id was returned in response

## Best Practices

1. **Run All Tests**: Use "Run All Tests" for comprehensive testing
2. **Check Results**: Review failed tests to identify issues
3. **Clean Database**: Start with a clean database for consistent results
4. **Verify Backend**: Ensure backend is running before testing
5. **Review Responses**: Expand response details to debug issues

## Future Enhancements

- [ ] Save test results to file
- [ ] Export test results as JSON
- [ ] Add test execution history
- [ ] Support custom test cases
- [ ] Add test performance metrics
- [ ] Support parallel test execution

