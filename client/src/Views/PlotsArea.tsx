import React, { useEffect, useState } from "react";
import { Col, Container, Row } from "reactstrap";
import { api, Method } from "../Api/Api";
import { IModel, inferenceIDs, TInferenceIDs } from "../Models/Model";
import { ModelSelector } from "./ModelSelector";
import { Plots } from "./Plots";
import { ActivePlots, PlotsSelector } from "./PlotsSelector";

export const dummy = "dummy";

export type Inferences = any;


const defaultActivePlots: ActivePlots = {} as ActivePlots;
for(const inferenceID of inferenceIDs)
{
    defaultActivePlots[inferenceID] = inferenceID != "s";
}

export function PlotsArea() {

    const [model, setModel] = useState<IModel>()
    const [inferences, setInferences] = useState<Inferences>()
    const [targets, setTargets] = useState<any>()
    const [activePlots, setActivePlots] = useState<ActivePlots>(defaultActivePlots)

    function handleModelChange(model: IModel)
    {
        setModel(model)
    }

    function handleActivePlotsChange(activePlots: ActivePlots)
    {
        setActivePlots(activePlots);
    }

    useEffect( () => {
            if(!model)
                return
        
            // fetch the inferences from new model
            const fetchInferences = async () => {
            const request = {
                method: Method.Get,
                url: `api/models/${model.model_name}/data/`,
                data: {}
            }
    
            const response = await api.callApi(request)
            setInferences(response.data.inferences)
            setTargets(response.data.targets)
        }
        
        fetchInferences()
    }, [model])

    const plotsSelector = model ? (
        <Row>
            <Col>
                <PlotsSelector onChange={handleActivePlotsChange} activePlots={activePlots}/>
            </Col>
        </Row>
    ) : null;

    const plots = inferences ? (
        <Row>
            <Col>
                <Plots inferences={inferences} targets={targets} activePlots={activePlots}/>
            </Col>
        </Row>
    ) : null;


    return(
        <Container>
            <Row>  
                <Col>
                    <ModelSelector onChange={handleModelChange}></ModelSelector>
                </Col>
            </Row>
            {
                plotsSelector
            }
            {
                plots
            }

        </Container>
    )

}