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

    const inferences = inferenceIDs.filter(id => props.activePlots[id]).map(inferenceID => props.inferences[inferenceID])
    const descriptors = inferenceIDs.filter(id => props.activePlots[id]).map(inferenceID => inferenceNameDictionary[inferenceID])

    const predictionLength = props.inferences[inferenceIDs[0]].length

    const chartData = []
    for(let i = 0; i < predictionLength; ++i)
    {
        const data = inferences.map( value => value[i])
        //console.log(data)
        chartData.push([i, ...data])
    }

    console.log(chartData)

    return (
        <Container>
                <Chart
                        chartType="LineChart"
                        data={[["Giorni", ...descriptors], ...chartData]}
                        width="100%"
                        height="400px"
                        legendToggle
                    />
        </Container>
    )
}