import { loadCss, loadModules } from "esri-loader";

class Mapping {
  webmapid = "e691172598f04ea8881cd2a4adaa45ba";
  mapView!: import("esri/views/MapView");

  async initializeMap(): Promise<void> {
    loadCss("https://js.arcgis.com/4.14/esri/css/main.css");

    // type the returned objects of loadModules using import types. Requires Typescript 2.9
    type MyModules = [
        typeof import("esri/Map"),
        typeof import("esri/WebMap"),
        typeof import("esri/views/MapView")
    ];
    const [Map, WebMap, MapView] = await (loadModules([
        "esri/Map",
        "esri/WebMap",
        "esri/views/MapView"
    ]) as Promise<MyModules>);
    // the returned objects now have type
    const map = new Map({
        basemap: "topo-vector"
    });

    // and we show that map in a container w/ id #viewDiv
    const view = new MapView({
      map: map,
      container: "viewDiv",
      center: [-118.80500, 34.02700],
      zoom: 13
    });

    view.when().then(() => {
      console.log("map is ready");
    });

    this.mapView = view;
  }
}

export default Mapping;
