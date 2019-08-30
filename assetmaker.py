import os, shutil, json
import xmlserializer
from copy import deepcopy
from modeler import Modeler
import csur
from csur import Segment

class AssetMaker:

    connectgroup = {'11': 'WideTram', '33': 'SingleTram', '44': 'NarrowTram',
                    '-31': 'DoubleTrain', '00': 'SingleTrain', '-2-2': 'TrainStation',
                    '13': 'CenterTram', '14': 'CenterTram'}

    names = {'g': 'basic', 'e': 'elevated', 'b': 'bridge', 't': 'tunnel', 's': 'slope'}
    shaders = {'g': 'Road', 'e': 'RoadBridge', 'b': 'RoadBridge', 't': 'Metro', 's': 'Metro'}
    suffix = {'e': 'express', 'w': 'weave'}
    textype = {'l': 'adr', 'g': 'd', 'e': 'dr', 'b': 'adrs', 't': 'd', 's': 'd', 'n': 'adr'}

    segment_presets = {}
    node_presets = {}
    lanes = {}
    props = {}

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
        with open(os.path.join(self.template_path, 'lanes.json'), 'r') as f:
            self.lanes = json.load(f)
        with open(os.path.join(self.template_path, 'props.json'), 'r') as f:
            self.props = json.load(f)
        self.dcnode_created = False

    def __initialize_assetinfo(self, asset):
        self.assetdata = {}
        self.assetdata['name'] = str(asset)
        for v in AssetMaker.names.values():
            with open(os.path.join(self.template_path, 'netinfo', '%s.json' % v), 'r') as f:
                jsondata = json.load(f)
                self.assetdata[v] = jsondata
            with open(os.path.join(self.template_path, 'net_ai', '%s.json' % v), 'r') as f:
                jsondata = json.load(f)
                self.assetdata['%sAI' % v] = jsondata
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

    def __add_node(self, name, mode='g', preset='default', color=[0.5, 0.5, 0.5], connectgroup=None):
        newmesh = self.__create_mesh(color, AssetMaker.shaders[mode[0]], name)
        modename = AssetMaker.names[mode[0]]
        self.assetdata['%sModel' % modename]['nodeMeshes']['CSMesh'].append(newmesh)
        nodeinfo = deepcopy(self.node_presets[preset])
        self.assetdata[modename]['m_nodes']['Node'].append(nodeinfo)
        if connectgroup:
            self.assetdata[modename]['m_nodes']['Node'][-1]['m_connectGroup'] = connectgroup


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
            shutil.copy(src, os.path.join(self.output_path, '%s_%s.png' % (name, t)))

    def __write_dcnode_textures(self, name):
        for t in AssetMaker.textype['l']:
            src = os.path.join(self.texture_path, 'lane_%s.png' % t)
            shutil.copy(src, os.path.join(self.output_path, '%s_%s.png' % (name, t)))

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

    def __create_node(self, asset):
        seg = asset.get_model('g')
        name = str(seg) + '_node'
        pavement, junction = self.modeler.make_node(seg)
        self.modeler.save(pavement, os.path.join(self.output_path, '%s_pavement.FBX' % name))
        self.__add_node('%s_pavement' % name, preset='default')
        self.__write_node_textures('%s_pavement' % name)
        self.modeler.save(junction, os.path.join(self.output_path, '%s_junction.FBX' % name))
        self.__add_node('%s_junction' % name, preset='trafficlight')
        self.__write_node_textures('%s_junction' % name)

    def __create_dcnode(self, asset, target_median='11'):
        MW = 1.875
        seg = asset.get_model('g')
        name = '%s_dcnode_%s' % (str(seg), target_median)
        split = 1 if target_median[0] != '-' else 2
        medians = [-int(target_median[:split])*MW, int(target_median[split:])*MW]
        lanes, side = self.modeler.make_dc_node(seg, target_median=medians)
        self.modeler.save(lanes, os.path.join(self.output_path, '%s_lanes.FBX' % name))
        self.__add_node('%s_lanes' % name, preset='direct', connectgroup=AssetMaker.connectgroup[target_median])
        self.__write_dcnode_textures('%s_lanes' % name)
        if not self.dcnode_created:
           # self.modeler.save(side, os.path.join(self.output_path, '%s_side.FBX' % name))
           # self.__add_node('%s_side' % name, preset='nojunction')
           # self.__write_dcnode_textures('%s_side' % name)
           self.dcnode_created = True

    def __create_lanes(self, seg, mode, reverse=False):
        if isinstance(seg, csur.TwoWay):
            self.__create_lanes(seg.left, mode, reverse=True)
            self.__create_lanes(seg.right, mode, reverse=False)
        else:
            modename = AssetMaker.names[mode[0]]
            for i, zipped in enumerate(zip(seg.start, seg.end)):
                u_start, u_end = zipped
                lane = None
                if u_start == u_end:
                    pos_start = (seg.x_start[i] + seg.x_start[i + 1]) / 2
                    pos_end = (seg.x_end[i] + seg.x_end[i + 1]) / 2
                    pos = (pos_start + pos_end) / 2
                    if reverse:
                        pos = -pos
                    if u_start == Segment.LANE:
                        lane = deepcopy(self.lanes['car'])
                    elif u_start == Segment.BIKE:
                        lane = deepcopy(self.lanes['bike'])
                    elif u_start == Segment.SIDEWALK:
                        lane = deepcopy(self.lanes['ped'])
                    if lane is not None:
                        lane['m_position'] = str(pos)
                        if u_start != Segment.SIDEWALK:
                            if reverse:
                                lane['m_direction'] = 'Backward'
                                lane['m_finalDirection'] = 'Backward'
                            else:
                                lane['m_direction'] = 'Forward'
                                lane['m_finalDirection'] = 'Forward'
                        self.assetdata[modename]['m_lanes']['Lane'].append(lane)
                    

                

    def __write_netAI(self, seg, mode):
        modename = AssetMaker.names[mode[0]]
        if mode[0] == 'g':
            self.assetdata['%sAI' % modename]['m_trafficLights'] = 'true'

    def __write_info(self, seg, mode):
        modename = AssetMaker.names[mode[0]]
        info = self.assetdata[modename]
        info["m_halfWidth"] = "%.3f" % min([max(seg.right.x_start), max(seg.left.x_start)])

    def writetoxml(self, asset):
        #path = os.path.join(self.output_path, str(asset.get_model('g')) + '.xml')
        path = os.path.join(self.output_path, 'road.xml')
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
        if asset.is_twoway and asset.roadtype == 'b':
            self.__create_node(asset)
            self.__create_dcnode(asset, target_median='11')
            #self.__create_dcnode(asset, target_median='33')
        # write data
        for mode in modes:
            seg = asset.get_model(mode)
            self.__create_lanes(seg, mode)
            self.__write_netAI(seg, mode)
            self.__write_info(seg, mode)
        self.writetoxml(asset)
        return self.assetdata