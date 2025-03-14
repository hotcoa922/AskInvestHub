// src/components/QuestionForm.js
import React, { useState } from 'react';
import axios from 'axios';
import './QuestionForm.css';

const QuestionForm = ({ onResponse }) => {
  const [query, setQuery] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;
    try {
      // Docker 컨테이너에서 실행 중인 백엔드 주소에 맞게 URL을 설정합니다.
      const response = await axios.post('http://localhost:8000/agent/ask', { query });
      onResponse(response.data);
    } catch (error) {
      console.error('API 호출 오류:', error);
      onResponse({ error: '서버와 통신 중 오류가 발생했습니다.' });
    }
  };

  return (
    <form onSubmit={handleSubmit} className="question-form">
      <input
        type="text"
        placeholder="질문을 입력하세요"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        className="question-input"
      />
      <button type="submit" className="submit-button">
        질문 전송
      </button>
    </form>
  );
};

export default QuestionForm;
