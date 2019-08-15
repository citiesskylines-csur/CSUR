import os, shutil, json
import xmlserializer
from copy import deepcopy
from modeler import Modeler

class AssetMaker:
    names = {'g': 'basic', 'e': 'elevated', 'b': 'bridge', 't': 'tunnel', 's': 'slope'}
    shaders = {'g': 'Road', 'e': 'RoadBridge', 'b': 'RoadBridge', 't': 'Metro', 's': 'Metro'}
    suffix = {'e': 'express', 'w': 'weave'}
    textype = {'l': 'adr', 'g': 'd', 'e': 'dr', 'b': 'adrs', 't': 'd', 's': 'd', 'n': 'adr'}

    segment_presets = {}
    node_presets = {}

    def __init__(self, dir, config_file='csur_blender.ini', texture_path='textures', 
                 template_path='templates', output_path='output', bridge=False, tunnel=True):
        self.modeler = Modeler(os.path.join(dir, config_file), bridge, tunnel)
        self.output_path = os.path.join(dir, output_path)
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        self.texture_path = os.path.join(dir, texture_path)
        self.template_path = os.path.join(dir, template_path)
        self.bridge = bridge
        self.tunnel = tunnel
        self.assetdata = {}
        with open(os.path.join(self.template_path, 'segment_presets.json'), 'r') as f:
            self.segment_presets = json.load(f)
        with open(os.path.join(self.template_path, 'node_presets.json'), 'r') as f:
            self.node_presets = json.load(f)

    def __initialize_assetinfo(self, asset):
        self.assetdata = {}
        self.assetdata['name'] = str(asset)
        with open(os.path.join(self.template_path, 'netinfo.json'), 'r') as f:
            jsondata = json.load(f)
            for v in AssetMaker.names.values():
                self.assetdata[v] = deepcopy(jsondata)
        with open(os.path.join(self.template_path, 'netAI.json'), 'r') as f:
            jsondata = json.load(f)
            for v in AssetMaker.names.values():
                self.assetdata['%sAI' % v] = deepcopy(jsondata)
        for v in AssetMaker.names.values():
            self.assetdata['%sModel' % v] = {'segmentMeshes': {'CSMesh': []}, 'nodeMeshes': {'CSMesh': []}}
        return self.assetdata

    def __create_mesh(self, color, shader, name):
        color = {'float': [str(x) for x in color]}
        csmesh = {}
        csmesh['color'] = color
        csmesh['shader'] = 'Custom/Net/%s' % shader
        csmesh['name'] = name
        return csmesh

    def __add_segment(self, name, mode='g', preset='default', color=[0.5, 0.5, 0.5]):
        newmesh = self.__create_mesh(color, AssetMaker.shaders[mode[0]], name)
        modename = AssetMaker.names[mode[0]]
        self.assetdata['%sModel' % modename]['segmentMeshes']['CSMesh'].append(newmesh)
        segmentinfo = deepcopy(self.segment_presets[preset])
        self.assetdata[modename]['m_segments']['Segment'].append(segmentinfo)

    def __add_node(self, name, mode='g', preset='default', color=[0.5, 0.5, 0.5]):
        newmesh = self.__create_mesh(color, AssetMaker.shaders[mode[0]], name)
        modename = AssetMaker.names[mode[0]]
        self.assetdata['%sModel' % modename]['nodeMeshes']['CSMesh'].append(newmesh)
        nodeinfo = deepcopy(self.node_presets[preset])
        self.assetdata[modename]['m_nodes']['Node'].append(nodeinfo)


    def __write_lane_textures(self, mode, name, split=False):
        for t in AssetMaker.textype['l']:
            src = os.path.join(self.texture_path, 'lane_%s.png' % t)
            if split:
                shutil.copy(src, os.path.join(self.output_path, '%s_%slanes_f_%s.png' % (name, mode, t)))
                shutil.copy(src, os.path.join(self.output_path, '%s_%slanes_r_%s.png' % (name, mode, t)))
            else:
                shutil.copy(src, os.path.join(self.output_path, '%s_%slanes_%s.png' % (name, mode, t)))

    def __write_struc_textures(self, mode, name):
        modename = AssetMaker.names[mode[0]]
        for t in AssetMaker.textype[mode[0]]:
            src = os.path.join(self.texture_path, '%s_%s.png' % (AssetMaker.names[mode[0]], t))
            shutil.copy(src, os.path.join(self.output_path, '%s_%s_%s.png' % (name, modename, t)))

    def __write_node_textures(self, name):
        for t in AssetMaker.textype['n']:
            src = os.path.join(self.texture_path, 'node_%s.png' % t)
            shutil.copy(src, os.path.join(self.output_path, '%s_node_%s.png' % (name, t)))

    def __create_segment(self, asset, mode):
        modename = AssetMaker.names[mode[0]]
        seg = asset.get_model(mode)
        name = str(seg)
        # make model
        seg_lanes, seg_struc = self.modeler.make(seg, mode)
        if len(mode) > 1:
            modename += AssetMaker.suffix[mode[1]]
        # save model and textures
        if asset.is_twoway() and asset.roadtype == 'r':
            self.modeler.save(seg_lanes[0], os.path.join(self.output_path, '%s_%slanes_f.FBX' % (name, mode)))
            self.modeler.save(seg_lanes[1], os.path.join(self.output_path, '%s_%slanes_r.FBX' % (name, mode)))
            self.__add_segment('%s_%slanes_f' % (name, mode), mode=mode[0])
            self.__add_segment('%s_%slanes_r' % (name, mode), mode=mode[0])
            self.__write_lane_textures(mode, name, split=True)
        else:
            self.modeler.save(seg_lanes, os.path.join(self.output_path, '%s_%slanes.FBX' % (name, mode)))
            self.__add_segment('%s_%slanes' % (str(seg), mode), mode=mode[0])
            self.__write_lane_textures(mode, name)
        self.modeler.save(seg_struc, os.path.join(self.output_path, '%s_%s.FBX' % (name, modename)))
        self.__add_segment('%s_%s' % (str(seg), modename), mode=mode[0])
        self.__write_struc_textures(mode, name)

    def __create_node(self, asset, preset='default'):
        seg = asset.get_model('g')
        name = str(seg)
        if preset != 'default':
            name += '_' + preset
        node = self.modeler.make_node(seg)
        self.modeler.save(node, os.path.join(self.output_path, '%s_node.FBX' % name))
        self.__add_node('%s_node' % str(seg), preset=preset)
        self.__write_node_textures(name)

    def __create_lanes(self, seg, mode):
        modename = AssetMaker.names[mode[0]]
        pass

    def __write_netAI(self, seg, mode):
        modename = AssetMaker.names[mode[0]]
        pass

    def __write_info(self, seg, mode):
        modename = AssetMaker.names[mode[0]]
        info = self.assetdata[modename]
        info["m_halfWidth"] = "%.3f" % max(seg.right.x_start)
        if mode[0] in 'gst':
            info["m_flattenTerrain"] = "true"
            info["m_clipTerrain"] = "true"
        else:
            info["m_flattenTerrain"] = "false"
            info["m_clipTerrain"] = "false"
            info["m_createPavement"] = "false"

    def writetoxml(self, asset):
        path = os.path.join(self.output_path, str(asset.get_model('g')) + '.xml')
        xmlserializer.write(self.assetdata, 'RoadAssetInfo', path)
    
    def make(self, asset, weave=False):
        self.__initialize_assetinfo(asset)
        modes = ['g', 'e']
        if self.tunnel:
            modes.append('t')
        if weave:
            modes = [x + 'w' for x in modes]
        if asset.roadtype == 'b':
            if self.bridge:
                modes.append('b')
            if self.tunnel:
                modes.append('s')
        # build segments
        for mode in modes:
            self.__create_segment(asset, mode)
        # build node
        if asset.is_twoway:
            self.__create_node(asset)
        # write data
        for mode in modes:
            seg = asset.get_model(mode)
            self.__create_lanes(seg, mode)
            self.__write_netAI(seg, mode)
            self.__write_info(seg, mode)
        self.writetoxml(asset)
        return self.assetdata