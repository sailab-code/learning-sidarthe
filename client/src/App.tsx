import React, { useEffect, useState } from 'react';
import './App.css';
import { Header } from './Views/Header';
import { Router, Route, Switch, Redirect } from "react-router-dom";
import { createBrowserHistory } from 'history';
import { Home } from './Views/Home';

function App() {
  const hist = createBrowserHistory();

	return (
    <React.Fragment>
      <Header />
      <br />
      <Router history={hist}>
        <Switch>
          <Route path="/" component={Home} />
        </Switch>
      </Router>
    </React.Fragment> 
	)
}

export default App;
