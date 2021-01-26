Compartmental Models
=====================

All the models inherits from CompartmentalModel an abstract class that extends pl.LightningModule.

CompartmentalModel
-------------------

.. autoclass:: lcm.compartmental_model.CompartmentalModel
	:members:

SIR
----

.. autoclass:: lcm.sir.SIR
	:members:

SIDARTHE
---------

.. image:: ../figs/sidarthe.png
	:width: 500
	
.. autoclass:: lcm.sidarthe.Sidarthe
	:members:
	

	

Mobility SIDARTHE
------------------

.. autoclass:: lcm.mobility_sidarthe.SidartheMobility
	:members:


Create a Custom Model
----------------------

Creating a custom compartmental model is easy. It is enough to extend the :class:`~lcm.compartmental_model.CompartmentalModel` class.

 .. code-block:: python
 
	from lcm.compartmental_model import CompartmentalModel 
	
	...
	
	class MyCustomCompartmentalModel(CompartmentalModel):
		def __init__():
			pass
		
		def differential_equations():
			pass