import React from "react";
import { Container, Col, Row } from "reactstrap";
import { Chart } from "react-google-charts"
import { SelectableChart, IData } from "./SelectableChart";
import { AreaTabView } from "./AreaTabView";

const deceased = [[1, 6], [2, 9], [3, 9], [4, 14], [5, 17], [6, 23], [7, 24], [8, 38], [9, 55], [10, 73], [11, 98], [12, 135], [13, 154], [14, 267], [15, 333], [16, 468], [17, 617], [18, 744], [19, 890], [20, 966], [21, 1218], [22, 1420], [23, 1640], [24, 1959], [25, 2168], [26, 2549], [27, 3095], [28, 3456], [29, 3776]];
const total = [[1, 172], [2, 240], [3, 258], [4, 403], [5, 531], [6, 615], [7, 984], [8, 1254], [9, 1520], [10, 1820], [11, 2251], [12, 2612], [13, 3420], [14, 4189], [15, 5469], [16, 5791], [17, 7280], [18, 8725], [19, 9820], [20, 11685], [21, 13272], [22, 14649], [23, 16220], [24, 17713], [25, 19884], [26, 22264], [27, 25515], [28, 27206], [29, 28761]]; 


const dataChoices: IData[] = [
    {
        descriptor: "Deceduti",
        data: deceased
    },
    {
        descriptor: "Casi Totali",
        data: total
    }
]

export function Home() {

    return (
        <Container>
            <Row>
                <Col>
                    <SelectableChart dataChoices={dataChoices} />
                </Col>
            </Row>

        </Container>
    )

}