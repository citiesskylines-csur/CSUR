import os, shutil, json
import xmlserializer
from copy import deepcopy
from modeler import ModelerLodded
import csur
from csur import Segment
from csur import StandardWidth as SW
from thumbnail import draw
import prop_utils

class AssetMaker:

    connectgroup = {'None': 'None', '11': 'WideTram', '33': 'SingleTram', '31': 'NarrowTram',
                    '3-1': 'DoubleTrain', '00': 'CenterTram', '1-1': 'SingleTrain', 'other': 'TrainStation'}
    

    # note: metro mode is used to visualize underground construction
    names = {'g': 'basic', 'e': 'elevated', 'b': 'bridge', 't': 'tunnel', 's': 'slope'}
    shaders = {'g': 'Road', 'e': 'RoadBridge', 'b': 'RoadBridge', 't': 'RoadBridge', 's': 'RoadBridge'}
    suffix = {'e': 'express', 'w': 'weave', 'c': 'compact', 'p': 'parking'}
    
    segment_presets = {}
    node_presets = {}
    lanes = {}
    props = {}

    def __init__(self, dir, config_file='csur_blender.ini',
                 template_path='templates', output_path='output', bridge=False, tunnel=True):
        self.modeler = ModelerLodded(os.path.join(dir, config_file), bridge, tunnel)
        self.output_path = os.path.join(dir, output_path)
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        self.template_path = os.path.join(dir, template_path)
        self.workdir = dir
        self.bridge = bridge
        self.tunnel = tunnel
        self.assetdata = {}
        self.assets_made = []
        with open(os.path.join(self.template_path, 'segment_presets.json'), 'r') as f:
            self.segment_presets = json.load(f)
        with open(os.path.join(self.template_path, 'node_presets.json'), 'r') as f:
            self.node_presets = json.load(f)
        with open(os.path.join(self.template_path, 'skins.json'), 'r') as f:
            self.skins = json.load(f)
        with open(os.path.join(self.template_path, 'props.json'), 'r') as f:
            self.props = json.load(f)
        self.lanes = {}
        for path in os.listdir(os.path.join(self.template_path, 'lane')):
            with open(os.path.join(self.template_path, 'lane', path), 'r') as f:
                self.lanes[os.path.splitext(path)[0]] = json.load(f)

    def get_connectgroup(self, key):
        if key in AssetMaker.connectgroup:
            return AssetMaker.connectgroup[key]
        else:
            return AssetMaker.connectgroup['other']

    def __initialize_assetinfo(self, asset):
        self.assetdata = {}
        self.assetdata['name'] = str(asset.get_model('g'))
        for v in AssetMaker.names.values():
            with open(os.path.join(self.template_path, 'netinfo', '%s.json' % v), 'r') as f:
                jsondata = json.load(f)
                self.assetdata[v] = jsondata
            with open(os.path.join(self.template_path, 'net_ai', '%s.json' % v), 'r') as f:
                jsondata = json.load(f)
                self.assetdata['%sAI' % v] = jsondata
            self.assetdata['%sModel' % v] = {'segmentMeshes': {'CSMesh': []}, 'nodeMeshes': {'CSMesh': []}}
        return self.assetdata

    def __create_mesh(self, color, shader, name, tex=None):
        color = {'float': [str(x) for x in color]}
        csmesh = {}
        csmesh['color'] = color
        csmesh['shader'] = 'Custom/Net/%s' % shader
        csmesh['name'] = name
        if tex == 'disabled':
            csmesh['texture'] = ''
        else:
            csmesh['texture'] = tex or name.split('_')[-1]
        return csmesh

    def __add_segment(self, name, model, mode='g', texmode=None, preset='default', color=[0.5, 0.5, 0.5]):
        self.modeler.save(model, os.path.join(self.output_path, name + '.FBX'))
        modename = AssetMaker.names[mode[0]]
        texmode = texmode or modename
        if texmode == 'metro':
            newmesh = self.__create_mesh(color, 'Metro', name, 'disabled')
        else:
            newmesh = self.__create_mesh(color, AssetMaker.shaders[mode[0]], name, texmode)
        self.assetdata['%sModel' % modename]['segmentMeshes']['CSMesh'].append(newmesh)
        segmentinfo = deepcopy(self.segment_presets[preset])
        self.assetdata[modename]['m_segments']['Segment'].append(segmentinfo)

    def __add_node(self, name, model, mode='g', texmode=None, preset='default', color=[0.5, 0.5, 0.5], connectgroup=None):
        self.modeler.save(model, os.path.join(self.output_path, name + '.FBX'))
        modename = AssetMaker.names[mode[0]]
        texmode = texmode or modename
        newmesh = self.__create_mesh(color, AssetMaker.shaders[mode[0]], name, texmode)     
        self.assetdata['%sModel' % modename]['nodeMeshes']['CSMesh'].append(newmesh)
        nodeinfo = deepcopy(self.node_presets[preset])
        self.assetdata[modename]['m_nodes']['Node'].append(nodeinfo)
        if connectgroup:
            self.assetdata[modename]['m_nodes']['Node'][-1]['m_connectGroup'] = connectgroup

    def __create_segment(self, asset, mode):
        modename = AssetMaker.names[mode[0]]
        seg = asset.get_model(mode)
        name = str(seg)
        # make model
        ret = self.modeler.make(seg, mode)
        if len(ret) == 2:
            seg_lanes, seg_struc = ret
        else:
            seg_lanes_f, seg_lanes_r, seg_struc = ret
        if len(mode) > 1:
            modename += AssetMaker.suffix[mode[1]]
        if asset.is_twoway() and asset.roadtype=='b' and asset.center()[0] == 0:
            preset_lane = 'default' if asset.left.nl() == asset.right.nl() else 'default_asym'
            preset_struc = 'default'
        else:
            preset_lane = preset_struc = 'default_nostop'
        if mode[0] == 's':
                preset_lane = preset_struc = 'slope'
        # save model and textures
        if asset.is_twoway() and asset.roadtype == 'r':
            self.__add_segment('%s_%slanes_f' % (name, mode), seg_lanes_f, mode=mode[0], preset=preset_lane, texmode='lane')
            self.__add_segment('%s_%slanes_r' % (name, mode), seg_lanes_r, mode=mode[0], preset=preset_lane, texmode='lane')
        else:
            self.__add_segment('%s_%slanes' % (name, mode), seg_lanes, mode=mode[0], preset=preset_lane, texmode='lane')
        self.__add_segment('%s_%s' % (name, modename), seg_struc, mode=mode[0], preset=preset_struc, texmode='tunnel' if mode[0] == 's' else None)
        if mode[0] == 't':
            arrows = self.modeler.make_arrows(seg)
            self.__add_segment('%s_arrows' % name, arrows, mode='t', texmode='metro')

    def __create_stop(self, asset, mode, busstop):
        if not busstop:
            raise ValueError("stop type should be specified!")
        modename = AssetMaker.names[mode[0]]
        seg = asset.get_model(mode)
        name = str(seg) + bool(busstop) * '_stop_%s' % busstop
        if busstop == 'brt':
            seg_lanes, seg_struc, brt_f, brt_both = self.modeler.make(seg, mode, busstop=busstop)
            preset = 'default_nostop'
        else:
            seg_lanes, seg_struc = self.modeler.make(seg, mode, busstop=busstop)
            preset = 'stop' + busstop
        if len(mode) > 1:
            modename += AssetMaker.suffix[mode[1]]
        self.__add_segment('%s_%slanes' % (name, mode), seg_lanes, mode=mode[0], preset=preset, texmode='lane')
        self.__add_segment('%s_%s' % (name, modename), seg_struc, mode=mode[0], preset=preset)
        if busstop == 'brt':
            self.__add_segment('%s_brt_single' % name, brt_f, mode=mode[0], preset='stopsingle', texmode='brt_platform')
            self.__add_segment('%s_brt_double' % name, brt_both, mode=mode[0], preset='stopdouble', texmode='brt_platform')


    def __create_node(self, asset):
        seg = asset.get_model('g')
        name = str(seg) + '_node'
        sidewalk, sidewalk2, asphalt, junction = self.modeler.make_node(seg)
        sidewalk_comp, asphalt_comp = self.modeler.make_node(seg, compatibility=True)
        self.__add_node('%s_sidewalk_crossing' % name, sidewalk, preset='trafficlight_nt', texmode='node')
        self.__add_node('%s_sidewalk_nocrossing' % name, sidewalk2, preset='notrafficlight_nt', texmode='node')
        self.__add_node('%s_asphalt' % name, asphalt, preset='default', texmode='node')
        self.__add_node('%s_junction' % name, junction, preset='trafficlight', texmode='node')
        self.__add_node('%s_sidewalk_comp' % name, sidewalk_comp, preset='transition', texmode='node')
        self.__add_node('%s_asphalt_comp' % name, asphalt_comp, preset='transition', texmode='node')
    

    def __create_dcnode(self, asset, target_median=None, asym_mode=None):
        MW = 1.875
        seg = asset.get_model('g')
        if target_median is None:
            medians = None
            target_median = self.__get_mediancode(asset)
        else:
            split = 1 if target_median[0] != '-' else 2
            medians = [-int(target_median[:split])*MW, int(target_median[split:])*MW]
        if asym_mode != 'invert':
            if asym_mode == 'restore':
                dcnode, target_median = self.modeler.make_asym_restore_node(seg)
                print(target_median)
                name = '%s_restorenode' % str(seg)
            elif asym_mode == 'expand':
                dcnode, target_median = self.modeler.make_asym_invert_node(seg, halved=True) 
                print(target_median)
                name = '%s_expandnode' % str(seg)
            else:
                dcnode = self.modeler.make_dc_node(seg, target_median=medians)
                name = '%s_dcnode_%s' % (str(seg), target_median)
            self.__add_node(name, dcnode, preset='direct', connectgroup=self.get_connectgroup(target_median), texmode='lane')
        else:
            # note that "bend node" is actually a segment in the game
            asym_forward, asym_backward = self.modeler.make_asym_invert_node(seg, halved=False)
            self.__add_segment('%s_asymforward' % str(seg), asym_forward, mode='g', preset='asymforward', texmode='lane')
            self.__add_segment('%s_asymbackward' % str(seg), asym_backward, mode='g', preset='asymbackward', texmode='lane')
        
    def __create_brtnode(self, asset):
        if not asset.is_twoway():
            raise ValueError("BRT station should be created on two-way roads!")
        mode = 'g'
        seg = asset.get_model(mode)
        blocks = asset.right.get_all_blocks()[0]
        seg_l = csur.CSURFactory(mode=mode, roadtype='b').get(
                                    blocks[0].x_left, blocks[0].nlanes)
        seg_r = csur.CSURFactory(mode=mode, roadtype='s').get([
                                    blocks[1].x_left - SW.MEDIAN, blocks[1].x_left], blocks[1].nlanes)
        dc_seg = csur.CSURFactory.fill_median(seg_l, seg_r, 's')
        dc_seg = csur.TwoWay(dc_seg.reverse(), dc_seg)
        model = self.modeler.convert_to_dcnode(dc_seg, keep_bikelane=False)
        self.__add_node('%s_brtnode' % str(seg), model, 
                        preset='direct', 
                        connectgroup=self.get_connectgroup(self.__get_mediancode(asset)), 
                        texmode='lane')

    # TODO: change speed limits
    def __create_lanes(self, asset, mode, seg=None, reverse=False, brt=False):
        modename = AssetMaker.names[mode[0]]
        if asset.is_twoway() and not seg:
            seg = asset.get_model(mode)
            self.__create_lanes(asset, mode, seg=seg.left, reverse=True)
            if not asset.is_undivided() and asset.append_median:
                median_lane = deepcopy(self.lanes['median'])
                # add traffic lights and road lights to median, lane position is always 0 to let NS2 work
                median_pos = (min(seg.right.x_start[0], seg.right.x_end[0]))
                prop_utils.add_props(median_lane, median_pos, self.props["light_median"])
                if asset.has_trafficlight():
                    # wide median is used if the road is wider than 6L
                    if max(asset.get_dim()) > 6 * SW.LANE:
                        prop_set = self.props["intersection_widemedian"]
                        xl = -asset.left.xleft[0] + SW.CURB
                        xr = asset.right.xleft[0] - SW.CURB
                    else:
                        prop_set = self.props["intersection_median"]
                        xl = xr = median_pos
                    prop_utils.add_intersection_props(median_lane, xr, prop_set)
                    prop_utils.add_intersection_props(median_lane, xl, prop_utils.flip(prop_set))
                self.assetdata[modename]['m_lanes']['Lane'].append(median_lane)
            self.__create_lanes(asset, mode, seg=seg.right, reverse=False)
        else:
            # keeps a bus stop lane cache; if the segment is a BRT module
            # then caches the first lane, else caches the last lane
            if not seg:
                seg = asset.get_model(mode)
            shift_lane_flag = False
            di_start = di_end = 0
            busstop_lane = None
            for i, zipped in enumerate(zip(seg.start, seg.end)):
                u_start, u_end = zipped
                lane = None
                if u_start == u_end:
                    if not shift_lane_flag and seg.roadtype() == 's' and min(seg.n_lanes()) > 1 and u_start == Segment.LANE:
                        if seg.x_start[0] - seg.x_end[0] > SW.LANE / 2:
                            di_start = -1
                            shift_lane_flag = True
                            continue
                        elif seg.x_end[0] - seg.x_start[0] > SW.LANE / 2:
                            di_end = -1
                            shift_lane_flag = True
                            continue
                    if u_start != Segment.LANE:
                        di_start = di_end = 0
                        shift_lane_flag = False
                    pos_start = (seg.x_start[i + di_start] + seg.x_start[i + 1 + di_start]) / 2
                    pos_end = (seg.x_end[i + di_end] + seg.x_end[i + 1 + di_end]) / 2
                    pos = (pos_start + pos_end) / 2
                    if u_start == Segment.LANE:
                        lane = deepcopy(self.lanes['car'])
                        if not (busstop_lane and brt):
                            busstop_lane = lane
                        # change prop positions in the car lane
                        for p in lane["m_laneProps"]["Prop"]:
                            deltax = (pos_end - pos_start) * float(p["m_segmentOffset"]) / 2
                            p["m_position"]["float"][0] = str(float(p["m_position"]["float"][0]) + deltax)
                    elif u_start == Segment.BIKE:
                        lane = deepcopy(self.lanes['bike'])
                    elif u_start == Segment.SIDEWALK:
                        lane = deepcopy(self.lanes['ped'])
                        # add ped lane props, first we determine where the car lanes end
                        i_side = len(seg.start) - len(csur.CSURFactory.roadside[mode])
                        x_side = (seg.x_start[i_side] + seg.x_end[i_side]) / 2
                        # determine the location where props are placed
                        if seg.start[i_side] == Segment.MEDIAN:
                            prop_pos = x_side + SW.MEDIAN / 2 - pos
                        elif seg.start[i_side] == Segment.CURB:
                            prop_pos = x_side + SW.CURB - pos
                        elif seg.start[i_side] == Segment.PARKING:
                            prop_pos = x_side + SW.CURB + SW.PARKING - pos
                        else:
                            raise NotImplementedError
                        # add lights and trees
                        if seg.x_start[i_side] == seg.x_end[i_side]:
                            prop_utils.add_props(lane, prop_pos, self.props["light_side"])
                            prop_utils.add_props(lane, prop_pos, self.props["random_street_prop"])
                            prop_utils.add_props(lane, prop_pos, self.props["tree_side"])
                        else:
                            light = deepcopy(self.props["light_side"])
                            for p in light:
                                p["m_repeatDistance"] = "0"
                            prop_utils.add_props(lane, prop_pos, light)
                            # two trees at -0.33 and + 0.33
                            tree_zpos = [-0.33, 0.33]
                            for z in tree_zpos:
                                tree = deepcopy(self.props["tree_side"])
                                for t in tree:
                                    t["m_repeatDistance"] = "0"
                                    t["m_segmentOffset"] = str(z)
                                deltax = (seg.x_end[i_side] - seg.x_start[i_side]) * z
                                prop_utils.add_props(lane, prop_pos + deltax, tree)
                        # add intersection props
                        if asset.has_trafficlight():
                            prop_utils.add_intersection_props(lane, prop_pos, self.props["intersection_side"])
                            # railway crossings should always be placed on sidewalks
                            if seg.start[i_side] == Segment.MEDIAN:
                                prop_pos += SW.MEDIAN / 2 + SW.BIKE + SW.CURB
                            prop_utils.add_intersection_props(lane, prop_pos, self.props["railway_crossing"])
                        # add bus stop props
                        if asset.has_busstop():
                            prop_pos = (seg.x_start[-1] + seg.x_start[-2]) / 2 - pos
                            prop_utils.add_props(lane, prop_pos, self.props["busstop"])

                    elif mode[0] == 'e' and u_start == Segment.BARRIER:
                        # only use left barrier light if width >= 5L
                        if i > 0 or max(seg.width()) > 5 * SW.LANE:
                            pos_start = seg.x_start[-1 * (i != 0)]
                            pos_end = seg.x_end[-1 * (i != 0)]
                            pos = (pos_start + pos_end) / 2
                            lane = deepcopy(self.lanes['barrier'])
                            light = deepcopy(self.props["light_side"])
                            if i == 0:
                                light = prop_utils.flip(light)
                            if seg.x_start[-1 * (i != 0)] != seg.x_end[-1 * (i != 0)]:
                                for p in light:
                                    p["m_repeatDistance"] = "0"
                            prop_utils.add_props(lane, 0, light)
                    if lane is not None:
                        # non-median lanes should never have 0 position,
                        # otherwise it confuses NS2
                        if pos == 0:
                            pos = 0.05
                        lane["m_position"] = str(-pos if reverse else pos)
                        if reverse:
                            lane = prop_utils.flip_lane(lane)                                  
                        self.assetdata[modename]['m_lanes']['Lane'].append(lane)
            # applies stop offset
            if mode[0] == 'g':
                if not brt:
                    busstop_lane["m_stopOffset"] = "-3" if reverse else "3"
                else:
                    busstop_lane["m_stopOffset"] = "-0.3" if reverse else "0.3"

    def __get_mediancode(self, asset):
        if not asset.is_twoway():
            return 'None'
        medians = asset.n_central_median()
        return str(medians[0]) + str(medians[1])

    def __write_netAI(self, asset, mode):
        seg = asset.get_model(mode)
        modename = AssetMaker.names[mode[0]]
        if mode[0] == 'g' and asset.is_twoway() and asset.roadtype == 'b':
            self.assetdata['%sAI' % modename]['m_trafficLights'] = 'true' 

    def __write_info(self, asset, mode):
        seg = asset.get_model(mode)
        modename = AssetMaker.names[mode[0]]
        info = self.assetdata[modename]
        if type(seg) == csur.TwoWay:
            if asset.roadtype == 'b':
                info["m_connectGroup"] = self.get_connectgroup(self.__get_mediancode(asset))
            else:
                info["m_connectGroup"] = "None"
            halfwidth = min([max(seg.right.x_start), max(seg.left.x_start)])
            if seg.right.start[-1] == Segment.SIDEWALK:
                halfwidth -= 1.25
            if asset.roadtype == 'b':
                if asset.asym()[0] > 0:
                    halfwidth += asset.asym()[0] * 1e-5
                elif asset.n_central_median()[0] > 1:
                    halfwidth += (asset.n_central_median()[0] - 1) * 1e-6
            # change min corner offset, increase the size of intersections
            # for roads wider than 6L
            # only apply to base modules
            if mode[0] == 'g' and asset.roadtype == 'b':
                if min(asset.get_dim()) > 6 * SW.LANE:
                    scale = 1 + (min(asset.get_dim()) - 3 * SW.LANE) / (SW.LANE * 20)
                else:
                    scale = 0.8
                info["m_minCornerOffset"] = str(halfwidth * scale)
                # clips terrain when there is median
                if asset.append_median:
                    info["m_clipTerrain"] = "true"
                    info["m_flattenTerrain"] = "true"
                    info["m_createPavement"] = "true"
                else:
                    info["m_clipTerrain"] = "false"
                    info["m_flattenTerrain"] = "false"
                    info["m_createPavement"] = "false"
        else:
            info["m_connectGroup"] = "None"
            halfwidth = max([max(seg.x_start), max(seg.x_end)])
            if seg.start[-1] == Segment.SIDEWALK:
                halfwidth -= 1.25
            if mode[0] != 'g':
                halfwidth += 1
            info["m_createPavement"] = "false"
            # slope mode must flatten terrain
            if mode[0] == 's':
                info["m_flattenTerrain"] = "true"
            else:
                info["m_flattenTerrain"] = "false"
            info["m_clipTerrain"] = "false"
            info["m_enableBendingNodes"] = "false"
            info["m_clipSegmentEnds"] = "false"
            info["m_minCornerOffset"] = "0"
        info["m_halfWidth"] = "%.8f" % halfwidth
        if asset.roadtype == 'b':
            info["m_enableBendingSegments"] = "true"


    def __get_light(self, asset, position, mode):
        # median lights
        if position == "median":
            if mode == "g":
                if min(asset.get_dim()) > 4 * SW.LANE:
                    return self.skins['light']['median_gnd']
            elif mode == "e":
                if min(asset.get_dim()) > 10 * SW.LANE:
                    return self.skins['light']['median_elv']
        # side lights
        elif position == "side":
            if mode == "g":
                if asset.is_twoway() and asset.is_undivided():
                    return self.skins['light']['side_gnd_large']
                elif min(asset.get_dim()) > 10 * SW.LANE:
                    return self.skins['light']['side_gnd_large']
                # divided roads narrower than 5L does not need side light
                elif asset.is_twoway() and min(asset.get_dim()) < 6 * SW.LANE:
                    return None
                else:
                    return self.skins['light']['side_gnd_small']
            elif mode == "e":
                return self.skins['light']['side_elv']
        return None

    '''
    Positions of lights and trees are included in the lane
    templates. The method searches for them using an identifier
    in the prop/tree name 'CSUR@*:*' then replaces them with the
    asset name in the skins file. If the prop/tree should not exist,
    it will be removed from the lane.
    '''
    # TODO: change light positions for interface modules
    def __apply_skin(self, asset, mode):
        modename = AssetMaker.names[mode[0]] 
        for lane in self.assetdata[modename]['m_lanes']['Lane']:
            # lights or trees should always be placed on a line with direction BOTH
            if lane["m_direction"] == "Both":
                removed = []
                for i, prop in enumerate(lane["m_laneProps"]["Prop"]):
                    if prop["m_prop"] and prop["m_prop"][:5] == "CSUR@":
                        split = prop["m_prop"][5:].lower().split(':')
                        if split[0] == "LIGHT":
                            key = self.__get_light(asset, split[1], mode[0])             
                        elif split[0] == 'SIGN':
                            proplist = self.skins[split[0].lower()][split[1].lower()]
                            if not asset.is_twoway:
                                key = None
                            else:
                                nlane = asset.left.nl() if float(prop["m_angle"] < 0) else asset.right.nl()
                                key = proplist[min(len(proplist), nlane) - 1]
                        if not key:
                            removed.append(i)
                        else:
                            prop["m_prop"] = key
                    if prop["m_tree"] and prop["m_tree"][:9] == "CSUR@TREE":
                        split = prop["m_tree"][5:].lower().split(':')
                        tree = self.skins[split[0]][split[1]]
                        if not tree:
                            removed.append(i)
                        else:
                            prop["m_tree"] = tree
                for i in removed[::-1]:
                    lane["m_laneProps"]["Prop"].pop(i)
        # place pillars, only base module has pillars
        if mode[0] == 'e' and asset.roadtype == 'b':
            if asset.is_twoway() and not asset.is_undivided():
                pillar = self.skins['pillar']['twoway'][int(min(asset.get_dim()) / SW.LANE / 2) - 1]
            else:
                blocks = asset.get_all_blocks()[0]
                if blocks[0].x_left * blocks[-1].x_right < 0:
                    pillar = self.skins['pillar']['twoway'][int(min(asset.get_dim()) / SW.LANE / 2) - 1]
                else:
                    pillar = None
            self.assetdata['elevatedAI']['m_bridgePillarInfo'] = pillar

    def writetoxml(self, asset):
        path = os.path.join(self.output_path, str(asset.get_model('g')) + '_data.xml')
        xmlserializer.write(self.assetdata, 'RoadAssetInfo', path)

    def write_thumbnail(self, asset):
        path = os.path.join(self.output_path, str(asset.get_model('g')))
        draw(asset, os.path.join(self.workdir, 'img/color.ini'), path)
        for mode in ['disabled', 'hovered', 'focused', 'pressed']:
            draw(asset, os.path.join(self.workdir, 'img/color.ini'), path, mode=mode)
    
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
        # build node. centered roads only
        if asset.is_twoway() and asset.roadtype == 'b' and asset.center()[0] == 0:
            n_central_median = asset.n_central_median()
            self.__create_node(asset)
            if n_central_median[0] == n_central_median[1]:
                # only create DC node for >3 lanes
                if asset.nl() > 3:
                    self.__create_dcnode(asset)
                    if n_central_median[0] == 1:
                        self.__create_dcnode(asset, target_median='33')
                    if n_central_median[0] == 0:
                        self.__create_dcnode(asset, target_median='11')
                self.__create_stop(asset, 'g', 'single')
                self.__create_stop(asset, 'g', 'double')
            else:
                if asset.nl() > 3:
                    if n_central_median[0] + n_central_median[1] > 0:
                        self.__create_dcnode(asset)
                        self.__create_dcnode(asset, asym_mode='expand')
                    self.__create_dcnode(asset, asym_mode='invert')   
                    self.__create_dcnode(asset, asym_mode='restore')
        # write data
        for mode in modes:
            self.__create_lanes(asset, mode)
            self.__write_netAI(asset, mode)
            self.__write_info(asset, mode)
            if mode[0] in 'get':
                self.__apply_skin(asset, mode)
        self.writetoxml(asset)
        self.write_thumbnail(asset)
        self.assets_made.append(str(asset.get_model('g')))
        return self.assetdata

    def make_singlemode(self, asset, mode):
        self.__initialize_assetinfo(asset)
        self.__create_segment(asset, mode)
        if asset.is_twoway() and asset.roadtype == 'b':
            self.__create_node(asset)
        self.__create_lanes(asset, mode)
        self.__write_netAI(asset, mode)
        self.__write_info(asset, mode)
        if mode[0] in 'get':
            self.__apply_skin(asset, mode)
        self.writetoxml(asset)
        self.write_thumbnail(asset)
        return self.assetdata


    def make_brt(self, asset):
        self.__initialize_assetinfo(asset)
        self.__create_stop(asset, 'g', 'brt')
        self.__create_node(asset)
        self.__create_brtnode(asset)
        self.__create_lanes(asset, 'g')
        self.__write_netAI(asset, 'g')
        self.__write_info(asset, 'g')
        self.assetdata['basic']["m_connectGroup"] = "None"
        self.__apply_skin(asset, 'g')
        self.writetoxml(asset)
        self.write_thumbnail(asset)
        return self.assetdata

    def output_assets(self):
        with open(os.path.join(self.output_path, 'imports.txt'), 'w+') as f:
            f.writelines(["%s\n" % x for x in self.assets_made])