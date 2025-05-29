// App.js

import React, { useState } from "react";
import "./App.css";

import Home from "./components/Home";
import BikeSharing from "./components/BikeSharing";
import StudentPerformance from "./components/StudentPerformance";

function App() {
  const tabs = [
    { id: "home", label: "Home", Component: Home },
    { id: "bike", label: "Bike Sharing", Component: BikeSharing },
    { id: "stu", label: "Student Performance", Component: StudentPerformance },
  ];

  const [activeTab, setActiveTab] = useState("home");
  const ActiveComponent = tabs.find((t) => t.id === activeTab).Component;

  return (
    <div className="App">
      <nav className="App-nav">
        {tabs.map((t) => (
          <button
            key={t.id}
            className={t.id === activeTab ? "tab active" : "tab"}
            onClick={() => setActiveTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </nav>
      <main className="App-content">
        <ActiveComponent />
      </main>
    </div>
  );
}

export default App;
