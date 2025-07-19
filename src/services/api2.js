// src/services/api.js
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const optimizePrompt = async (data) => {
  const response = await api.post('/optimize', data);
  return response.data;
};

export const getPromptLibrary = async () => {
  const response = await api.get('/prompt-library');
  return response.data;
};

export const getSystemStats = async () => {
  const response = await api.get('/stats');
  return response.data;
};

export const getUserMemory = async () => {
  const response = await api.get('/user-memory');
  return response.data;
};
