import React, { useState } from "react";
import { Input, Container, Row, Col } from "reactstrap";
import Chart from "react-google-charts";


export interface IData {
    descriptor: string;

    data: number[][]
}

interface IProps {
    dataChoices: IData[]
}


export function SelectableChart(props: IProps) {

    const { dataChoices } = props;
    const [currentData, setCurrentData] = useState(dataChoices[0])



    function handleSelect(event: React.ChangeEvent<HTMLInputElement>)
    {
        console.log(dataChoices)
        const idx = Number(event.currentTarget.value)
        const newData = dataChoices[idx]
        console.log(newData)
        setCurrentData(newData)
    }



    return (
        <Container>
            <Row>
                <Col>
                    <Input type="select" onChange={handleSelect}>
                        {
                            dataChoices.map( (data, index) => 
                                <option key={index} value={index}>{data.descriptor}</option>
                            )
                        }
                    </Input>
                </Col>
            </Row>
            <Row>
                <Col>
                <Chart
                    chartType="LineChart"
                    data={[["Giorni", currentData.descriptor], ...currentData.data]}
                    width="100%"
                    height="400px"
                    legendToggle
                />
                </Col>
            </Row>
        </Container>
    )

}