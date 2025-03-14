import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, useNavigate, useLocation } from 'react-router-dom';
import QuestionForm from './components/QuestionForm';
import ResultPage from './components/ResultPage';
import './App.css';

const HomePage = () => {
  const navigate = useNavigate();

  const handleResponse = (res) => {
    // 응답 데이터를 상태로 넘기면서 결과 페이지로 이동
    navigate('/result', { state: res });
  };

  return (
    <div className="container">
      <div className="card">
        <h1>증권 AI 서비스</h1>
        <QuestionForm onResponse={handleResponse} />
      </div>
    </div>
  );
};

const ResultPageWrapper = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const res = location.state;

  const handleBack = () => {
    navigate('/');
  };

  // 만약 응답 데이터가 없다면 홈으로 리다이렉트
  if (!res) {
    navigate('/');
    return null;
  }

  return <ResultPage {...res} onBack={handleBack} />;
};

const App = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/result" element={<ResultPageWrapper />} />
      </Routes>
    </Router>
  );
};

export default App;
