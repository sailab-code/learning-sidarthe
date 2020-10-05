import React, { useEffect, useState } from "react";
import { FormGroup, Input, Label } from "reactstrap";
import { TInferenceIDs, inferenceIDs } from "../Models/Model";

export type ActivePlots = any;

interface IProps {
    onChange: (activePlots: ActivePlots) => void;
    activePlots: ActivePlots;
}



export function PlotsSelector(props:IProps)
{
    function handleChange(inferenceID: TInferenceIDs, event: React.ChangeEvent<HTMLInputElement>){
        const newState = Boolean(event.currentTarget.checked)
        props.onChange({
            ...props.activePlots,
            [inferenceID]: newState
        })
    }

    const checkboxes = inferenceIDs.map( (inferenceID, index) => 
        <FormGroup key={index} check inline>
            <Label check>
                <Input type="checkbox" defaultChecked={true}  onChange={handleChange.bind(this, inferenceID)}/>
                {inferenceID.toUpperCase()}
            </Label>
        </FormGroup>
    )

    return (
        <div>
            {checkboxes}
        </div>
    )
}