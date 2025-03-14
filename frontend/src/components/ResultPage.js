import React from 'react';
import './ResultPage.css';

const ResultPage = ({ selected_agent, result, error, onBack }) => {
  return (
    <div className="result-page">
      <h2>결과 페이지</h2>
      {error ? (
        <div className="error-message">{error}</div>
      ) : (
        <>
          <h3>선택된 에이전트: {selected_agent}</h3>
          <pre className="result-text">
            {typeof result === 'object' ? JSON.stringify(result, null, 2) : result}
          </pre>
        </>
      )}
      <button className="back-button" onClick={onBack}>
        뒤로가기
      </button>
    </div>
  );
};

export default ResultPage;
