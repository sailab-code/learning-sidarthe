import axios, { AxiosRequestConfig } from "axios";
import { IData } from "../Views/SelectableChart";

export interface IDataRequest {
    area: "nation" | "region" | "province";
    areaName: string;
}

export interface IDataResponse {
    data: IData[]
}

export enum Method {
    Get = "GET",
    Post = "POST",
    Put = "PUT",
    Delete = "DELETE",
}

export class Api {

    Api() {
        //axios.defaults.baseURL = baseUrl;
        axios.defaults.headers.common["Content-Type"] = "application/json";
        this.callApi = this.callApi.bind(this);
    }

    async callApi(request: any){     //: Promise<IResponse>
        const dataOrParams = request.method in [Method.Get] ? "params" : "data";

        const requestConfig : AxiosRequestConfig = {
            url: request.url,
            method: request.method,
            [dataOrParams]: request.data
        };
        
        return await axios.request(requestConfig);
    }

    async getData(request: IDataRequest)
    {
        return await this.callApi({
            url: `/api/getData`,
            method: Method.Get,
            data: request
        })
    }
}

export const api = new Api();
