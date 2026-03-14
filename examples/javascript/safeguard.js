import axios from 'axios';

// Configuration
const GUARDRAILED_GATEWAY_URL = 'http://localhost:8000';

async function runSafeguardExample() {
  console.log('Running safeguard examples');

  // 1. Safe Request
  console.log('Testing a safe message...');
  const safePayload = {
    messages: [{ role: 'user', content: 'This is a perfectly safe message.' }],
  };
  try {
    const response = await axios.post(`${GUARDRAILED_GATEWAY_URL}/safeguard`, safePayload);
    console.log(`Status Code: ${response.status}`);
    console.log('Response JSON:', JSON.stringify(response.data, null, 2));
    console.log('--> Expected: safety_code: 0 (SAFE)');
  } catch (error) {
    console.error('Error during safe request:', error.response?.data || error.message);
  }

  // 2. Block PII Request
  // Requires PII policy to be active with action: 0 (OVERRIDE)
  console.log('Testing a message with PII to be BLOCKED...');
  const piiPayload = {
    messages: [{ role: 'user', content: 'My email is test@example.com .' }],
  };
  try {
    const response = await axios.post(`${GUARDRAILED_GATEWAY_URL}/safeguard`, piiPayload);
    console.log(`Status Code: ${response.status}`);
    console.log('Response JSON:', JSON.stringify(response.data, null, 2));
    console.log('--> Expected: safety_code: 20 (PII_DETECTED), action: 0 (OVERRIDE)');
  } catch (error) {
    console.error('Error during PII block request:', error.response?.data || error.message);
  }
}

runSafeguardExample();