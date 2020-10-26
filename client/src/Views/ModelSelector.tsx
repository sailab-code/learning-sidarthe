import React, { useEffect, useState } from "react"
import { api, Method } from "../Api/Api"
import { IModel } from "../Models/Model"
import { Input } from "reactstrap";

interface IProps {
    onChange: (model: IModel) => void;
}


export function ModelSelector(props: IProps) {
    const [modelList, setModelList] = useState([] as IModel[])

    // fetch models list
    useEffect( () => {
        const fetchModels = async () => {
            const request = {
                method: Method.Get,
                url: "/api/models/",
                data: {}
            };

            const response = await api.callApi(request);
            const modelsData = response.data as IModel[];
            modelsData.sort((a, b) => a.train_size - b.train_size)
            setModelList(modelsData);
        }

        fetchModels();
    }, [])
    
    function handleChange(event: React.ChangeEvent<HTMLInputElement>)
    {
        const idx = Number(event.currentTarget.value);
        const newModel = modelList[idx];
        props.onChange(newModel);
    }

    const placeholder = modelList.length == 0 ? "Loading models": "Select a model";

    function getModelViewText(model: IModel)
    {
        return `${model.region} => Trained with ${model.train_size} days.`
    }

    return (
        <Input type="select" onChange={handleChange} placeholder={placeholder} defaultValue={"none"}>
            {
                [
                    <option key={-1} disabled value={"none"}> -- Select a model --</option>,
                    ...modelList.map( (model, index) => 
                        <option key={index} value={index}>{getModelViewText(model)}</option>
                    )
                ]
            }
        </Input>
    )
    

}