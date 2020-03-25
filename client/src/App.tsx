import React, { useEffect } from 'react';
import logo from './logo.svg';
import './App.css';
import { Chart } from "react-google-charts"


function App() {
	return (
		<div className={"my-pretty-chart-container"}>
        <Chart
          chartType="ScatterChart"
          data={[["Age", "Weight"], [4, 5.5], [8, 12]]}
          width="100%"
          height="400px"
          legendToggle
        />
      </div>
	)
}

export default App;
