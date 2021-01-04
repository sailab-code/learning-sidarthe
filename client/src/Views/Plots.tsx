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
    targets: any;
    region: string;
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


const colorsDictionary = {
    "s": 'Tomato',
    "i": "DodgerBlue",
    "d": "MediumSeaGreen",
    "a": "SlateBlue",
    "r": "Violet",
    "t": "Orange",
    "h_detected": "Purple",
    "e": "Black",
    "r0": "Teal" 
}

const initialDates = {
  "Italy": new Date(2020, 2, 24),
  "FR": new Date(2020, 3, 17)
}


export function Plots(props: IProps) {

    if(!props.targets || !props.inferences)
      return null;

    const inferenceFilter = id => props.activePlots[id];
    const targetFilter = id => props.activePlots[id] && ['d', 'r', 't', 'h_detected', 'e'].includes(id);


    const activeInferences = inferenceIDs.filter(inferenceFilter);
    const activeTargets = inferenceIDs.filter(targetFilter);
    
    const inferencesInfo = activeInferences.map(inferenceID => ({
      data: props.inferences[inferenceID],
      descriptor: inferenceNameDictionary[inferenceID], 
      color: colorsDictionary[inferenceID],
      style: [1, 0]
    }));


    const targetsInfo = activeTargets.map(inferenceID => ({
      data: props.targets[inferenceID],
      descriptor: inferenceNameDictionary[inferenceID]+ " (observed)", 
      color: colorsDictionary[inferenceID],
      style: [2, 2]
    }));

    const infos = [...inferencesInfo, ...targetsInfo];

    
    infos.sort((infoA, infoB) => infoA.descriptor.localeCompare(infoB.descriptor))

    const descriptors = infos.map(info => info.descriptor)
    const colors = infos.map(info => info.color)

    const predictionLength = props.targets['d'].length + 50

    const chartData = []
    
    const series = {}

    for(let i = 0; i < infos.length; ++i)
    {
      series[i] = { lineDashStyle: infos[i].style }
    }

    const firstDate = initialDates[props.region];
    for(let i = 0; i < predictionLength; ++i)
    {
        const date = new Date(firstDate.valueOf())
        date.setDate(date.getDate() + i)

        
        const data = [date, ...infos.map(info => info.data[i])]
        chartData.push(data)
    }

    console.log(chartData)

    const controlStart = new Date(firstDate.valueOf())
    controlStart.setDate(controlStart.getDate() + 30);

    const controlEnd = new Date(firstDate.valueOf())
    controlEnd.setDate(controlStart.getDate() + 130)

    return (
        <Container>
                <Chart
                        chartType="LineChart"
                        data={[["Giorni", ...descriptors], ...chartData]}
                        width="100%"
                        height="400px"
                        options={{
                          colors: [...colors],
                          title: `Predictions for ${props.region}`,
                          vAxis: {
                            title: "# People"
                          },
                          hAxis: {
                            title: "Date",
                            format: "MMMM yyyy"
                          },
                          series: series
                        }}
                        controls={[
                            {
                                controlType: 'ChartRangeFilter',
                                options: {
                                  filterColumnIndex: 0,
                                  ui: {
                                    chartType: 'LineChart',
                                    chartView: {
                                      columns: [0, 10]
                                    },
                                    chartOptions: {
                                      chartArea: { width: '90%', height: '50%' },
                                      hAxis: { baselineColor: 'none', format: "MMMM yyyy", 'textPosition': 'out'},

                                    },
                                  },
                                },
                                controlPosition: 'bottom',
                                controlWrapperParams: {
                                  state: {
                                    range: { start: controlStart, end: controlEnd },
                                  },
                                },
                            }

                        ]}
                        legendToggle
                    />
        </Container>
    )
}