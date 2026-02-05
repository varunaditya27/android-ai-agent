## Problem Statement
 
Build an AI-powered mobile automation agent that assists blind users in operating an Android device through natural language commands in an interface of your choice.
 
## Core Requirements
 
**Device Execution**
- Use a cloud-based Android service 
 
**Agent Capabilities**
- Navigate to and open apps
- Perform searches or operate other actions within these apps
- Extract and return results to the user
- Handle basic system actions (e.g., device restart)
- Handle authentication flows by prompting the user for credentials when login screens are detected
 
**Interface**
- Accept natural language input
- Return actionable results/responses
- Prompt user for input when required (credentials, OTPs, confirmations)
 
## Deliverables
 
- Functional agent with cloud device integration
- Setup instructions
- Demo covering: open app → authenticate if needed → search → return results
 
## Evaluation Criteria
 
- Accuracy of navigation and search execution
- Quality of result extraction
- Handling of authentication and interactive prompts
- Code clarity and extensibility
 
## Technical Considerations
 
- Screen element detection approach (accessibility tree, vision model, or hybrid)
- State management across multi-step actions
- Error recovery when UI changes or actions fail
- Secure handling of user credentials (don't log or persist)
 
## Sample Interaction
 
```
User: "Open ChatGPT and ask what's the capital of France"
Agent: Opens ChatGPT app → detects login screen
Agent: "Login required. Please provide your email."
User: "user@example.com"
Agent: "Enter your password."
User: "********"
Agent: Completes login → sends query → extracts response
Response: "The capital of France is Paris."
```
 
## Questions to Address in Submission
 
- How do you identify UI elements reliably?
- How do you detect and handle authentication screens?
- How do you handle failed actions or unexpected popups?
- What tradeoffs did you make for latency vs accuracy?