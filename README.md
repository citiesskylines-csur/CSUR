# Cities: Skylines Urban Road

Cities: Skylines Urban Road (CSUR) is a fully modular road asset framework for Cities: Skylines created by procedural content generation and asset packaging. CSUR has been released onto [the Steam Workshop](https://steamcommunity.com/sharedfiles/filedetails/?id=1959216109). 

![Sample Interchange](https://github.com/victoriacity/CSUR/blob/master/csur-sample.png)
*An example road interchange built by the developer in Cities:Skylines with CSUR. The CSUR code generates straight pieces of roads and imports them into Unity prefabs, and then the player uses the in-game UI to build road structures. The game handles the stretching and bending of meshes in each piece of road segment.*

## Introduction

Cities: Skylines Urban Road, or CSUR in short, is an asset/mod suite for Cities:Skylines providing unprecedented realism in road networks for city-building games. Under its hood is an offline procedural generation system based on a high-level road design API and Blender graphics backend. Therefore, the core of CSUR is a [Python package](https://github.com/citiesskylines-csur/CSUR) generating game assets (Unity prefabs), and [several plugins written in Unity/C#](https://github.com/citiesskylines-csur) were also developed to modify relevant base game logics and convert asset sources into serialized prefab files. CSUR has enjoyed exceptional reception from the Cities: Skylines community and gained more than 35,000 cumulative users on the Steam Workshop. 

In the past 5 years, *Cities: Skylines* has almost dominated the city-building genre on the PC platform by providing outstanding features and engaging gameplay with more than [11 million copies sold on PC and console platforms](https://twitter.com/CitiesSkylines/status/1237332408061624320). Another critical reason why *Cities: Skylines* has enjoyed such popularity is that it provides extensive support for community-contributed game assets and mods (plug-in code libraries). *Cities: Skylines* is also one of the games with the largest amount of community contents on the Steam Workshop.

## Background

Roads are one of the most important types of urban infrastructure and plays a key role in almost any city-building game. Nevertheless, due to performance optimization and balancing the depth of multiple aspects of city simulation, roads in the base game of *Cities: Skylines* are created in a relatively simplistic manner, omitting all markings at highway ramps and lane transitions (i,e., addition or reduction of a lane). This has motivated enthusiastic *Cities: Skylines* players to add details to roads in their city builds using community-contributed assets and mods. A highlight among such efforts was the original [CSUE](https://steamcommunity.com/workshop/filedetails/?id=1423096565)/[CSUR](https://steamcommunity.com/workshop/filedetails/?id=1206133771) road asset packs created by AmamIya. The key idea of these asset packs was to make different combinations of ramps and lane transitions into draggable road segment modules instead of treating them as intersections as in the base game. This allows all road markings to be modeled in each module and greatly increases the realism of road infrastructure in *Cities: Skylines*.

Originally, all game assets in CSUE/CSUR were modeled and textured manually with hundreds of hours spent on creating 3D models and packaging them using the Asset Editor interface of the game. By leveraging procedural content generation and automatic asset packaging, the consistency of quality in assets and efficiency can be improved at an unprecedented scale. For example, the 2000-asset package done by the current CSUR code should have taken several years to be created manually. This has led to the development of the CSUR software suite in this repository.

Being the largest road content collection ever created for the game, we believe that CSUR will make a profound influence on the *Cities: Skylines* community. Although developed for *Cities: Skylines*, application of CSUR may have a broader picture. With the flexibility and realism in simulated road infrastructure delivered by CSUR, it also has the potential to create procedually-generated simulation environments involving urban roads, such as synthetic data to train machine learning systems for autonomous vehicles. With city-building games delivering scenes of simulation quality, crowdsourcing urban simulation environments for AI applications will become possible.

## Design
The CSUR project mainly consists of the following components:

1. A high-level API [`core/`](https://github.com/citiesskylines-csur/CSUR/tree/master/core) for the configuration of a road asset. The ultimate goal of CSUR is to generate any possible road configuration present in real-world cities. Therefore, it needs to be able to describe the composition of an urban road in its fundamental data structure. Besides, it assigns each road asset a unique name which is both human-readable and can be readily compiled back into road configuration data.

2. A 3D graphics library [`modeling/`](https://github.com/citiesskylines-csur/CSUR/tree/master/modeling) which utilizes the Blender Python backend to procedurally generate road meshes. It can be potentially migrated to other graphics backends (e.g., [PyMesh](https://pymesh.readthedocs.io/en/latest/index.html)) as long as they support texture mapping and FBX I/O format.

3. A sub-package [`prefab/`](https://github.com/citiesskylines-csur/CSUR/tree/master/prefab) which generates the prefab property data for each road asset based on its configurations. It takes JSON templates encoding common properties among road assets, e.g., traffic lights should be spawn at intersections, and outputs XML files to be further imported through in-game code. It also provides a command-line interface for users to generate their own asset in case the static collection released is not sufficient.

4. A 2D graphics library [`graphics/`](https://github.com/citiesskylines-csur/CSUR/tree/master/graphics) based on [PyCairo](https://www.cairographics.org/pycairo/) to create thumbnail images to visualize the configuration and functionality of each road asset in UI sprites.

5. Scripts [`builder/`](https://github.com/citiesskylines-csur/CSUR/tree/master/builder) to search for valid road configurations and build the list of road assets to be imported into the game. 

6. In-game Unity code (shipped as DLL binary at [`bin/`](https://github.com/citiesskylines-csur/CSUR/tree/master/bin)) to automatically invoke Asset Editor sessions and method calls. 

The dependency structure of the system was designed to be as decoupled as possible, so that some components can be run even without Blender or Cairo backends, such as generating the list of road assets for Steam Workshop release can be done straight from the Python shell. The below figure depicts how these components in the CSUR package are organized.

![](/csur-system-design.png)



