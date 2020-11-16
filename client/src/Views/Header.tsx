import React, { useState } from "react"


import { 
    Collapse,
    NavbarToggler,
    NavbarBrand,
    Nav,
    NavItem,
    NavLink,
    UncontrolledDropdown,
    DropdownToggle,
    DropdownMenu,
    DropdownItem,
    NavbarText,
    Navbar 
} from "reactstrap";

export function Header() {
    const [isOpen, setOpen] = useState(false);

    function toggle() {
        setOpen(!isOpen)
    }
    
    return (
        <Navbar color="dark" dark={true}  expand="md">
            <NavbarBrand href="/">SAILab COVID-19 Predictor</NavbarBrand>
      </Navbar>
    )
}