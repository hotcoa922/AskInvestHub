// src/components/ResponseDisplay.js
import React from 'react';
import './ResponseDisplay.css';

const ResponseDisplay = ({ response }) => {
  if (!response) return null;

  const renderResult = () => {
    if (typeof response.result === 'object') {
      // 원하는 방식으로 객체 내용을 표시합니다.
      // 여기서는 JSON.stringify를 사용하여 문자열로 변환합니다.
      return JSON.stringify(response.result, null, 2);
    }
    return response.result;
  };

  return (
    <div className="response-display">
      {response.error ? (
        <p className="error-text">{response.error}</p>
      ) : (
        <>
          <h3 className="agent-title">선택된 에이전트: {response.selected_agent}</h3>
          <pre className="response-text">{renderResult()}</pre>
        </>
      )}
    </div>
  );
};

export default ResponseDisplay;
