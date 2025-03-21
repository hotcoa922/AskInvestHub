// src/components/Sidebar.js
import React from 'react';
import './Sidebar.css';

const Sidebar = () => {
  return (
    <div className="sidebar">
      <h3>Tip of the Day</h3>
      <ul>
        <li>효과적인 투자 계획 수립</li>
        <li>위험 분산을 위해 다양한 자산에 투자</li>
        <li>시장 동향을 지속적으로 모니터링</li>
      </ul>
      <h3>Quick Info</h3>
      <p>
        최신 증권 뉴스와 관련 법률 업데이트를 확인하세요.
      </p>
    </div>
  );
};

export default Sidebar;
