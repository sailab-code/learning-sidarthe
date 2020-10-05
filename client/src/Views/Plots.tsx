import React from "react";
import { Col, Container, Row } from "reactstrap";
import { IModel, inferenceIDs, TInferenceIDs } from "../Models/Model";
import { Inferences } from "./PlotsArea";
import { ActivePlots } from "./PlotsSelector";
import Chart from "react-google-charts";
import { number } from "prop-types";

interface IProps {
    activePlots: ActivePlots;
    inferences: Inferences;
}

const inferenceNameDictionary = {
    "s": "Susceptible Individuals",
    "i": "Asymptomatic Infected (Undetected)",
    "d": "Asymptomatic Infected (Detected)",
    "a": "Symptomatic Infected (Undetected)",
    "r": "Symptomatic Infected (Detected)",
    "t": "Acutely Symptomatic Infected (ICU)",
    "h_detected": "Healed Individuals",
    "e": "Deceased Individuals",
    "r0": "R0" 
}


export function Plots(props: IProps) {

    const plots = [];

    for(const inferenceID of inferenceIDs)
    {
        if(props.activePlots[inferenceID])
        {

            const descriptor = inferenceNameDictionary[inferenceID]

            const data = props.inferences[inferenceID] as number[];
            const chartData = data.map( (value, index) => [index, value])

            plots.push(
                <Row key={inferenceID}>
                    <Col>
                        <Chart
                            chartType="LineChart"
                            data={[["Giorni", descriptor], ...chartData]}
                            width="100%"
                            height="400px"
                            legendToggle
                        />
                    </Col>
                </Row>
            )
        }
    }


    return (
        <Container>
                {
                    plots
                }
        </Container>
    )



}