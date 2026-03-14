import OpenAI from 'openai';
import process from 'process';

// Make sure to set your OpenAI API key as an environment variable
// export OPENAI_API_KEY="sk-..."
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
if (!OPENAI_API_KEY) {
  throw new Error('Please set the OPENAI_API_KEY environment variable.');
}

// Point the OpenAI client to your Guardrailed Gateway instance
const openai = new OpenAI({
  apiKey: OPENAI_API_KEY,
  baseURL: 'http://localhost:8000/v1',
});

async function runOpenAIProxyExamples() {
  console.log('Running OpenAI Proxy examples');

  // 1. Safe Request
  console.log('Testing a safe request that should pass...');
  try {
    const completion = await openai.chat.completions.create({
      model: 'gpt-3.5-turbo',
      messages: [{ role: 'user', content: 'What is the capital of France?' }],
      max_tokens: 10,
    });
    console.log('Response:', completion.choices[0].message.content.trim());
    console.log("--> Expected: A valid response from OpenAI (e.g., 'Paris').");
  } catch (error) {
    console.error('An error occurred:', error);
  }

  // 2. Blocked Request
  // Requires PII policy to be active with action: 0 (OVERRIDE)
  console.log('Testing a request with PII to be BLOCKED...');
  try {
    const completion = await openai.chat.completions.create({
        model: "gpt-3.5-turbo",
        messages: [{ role: "user", content: "My email is test@example.com" }],
    });

    console.log('Blocked Message:', completion.choices[0].message.content);
    console.log('Finish Reason:', completion.choices[0].finish_reason);
    console.log("--> Expected: finish_reason: 'content_filter'");
    
  } catch (error) {
    console.error('An error occurred:', error);
  }
}

runOpenAIProxyExamples();