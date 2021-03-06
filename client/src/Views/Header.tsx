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
            <NavbarBrand href="/">SAILaB COVID-19 Predictor</NavbarBrand>
            <NavbarToggler onClick={toggle}/>
            <Collapse isOpen={isOpen} navbar>
                <Nav className="mr-auto" navbar>
                <NavItem>
                        <NavLink href="/why/">A cosa serve</NavLink>
                    </NavItem>
                    
                    <NavItem>
                        <NavLink href="/about-us/">Chi siamo?</NavLink>
                    </NavItem>
                </Nav>
            </Collapse>
      </Navbar>
    )
}