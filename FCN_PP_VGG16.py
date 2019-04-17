import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
'''
input_chn = 3
input_size = 473
'''

'''
Use VGG16-BN(5 stages) before pyramid pooling
'''


class basicConv(nn.Module):
    def __init__(self, nInput, nOutput, kerSize, stride, pad):
        super(basicConv, self).__init__()
        self.nInput = nInput
        self.nOutput = nOutput
        self.kerSize = kerSize
        self.stride = stride
        self.pad = pad

        self.conv = nn.Conv2d(self.nInput, self.nOutput, kernel_size=self.kerSize, stride=self.stride, padding=self.pad,
                              bias=True)
        self.bn = nn.BatchNorm2d(self.nOutput, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x):
        x = self.conv(x)
        conv_out = x
        x = self.bn(conv_out)
        x = F.relu(x)
        return x, conv_out


class bottleNeckConv(nn.Module):
    def __init__(self, nInput, nOutput, kerSize=1, stride=1, pad=0):
        super(bottleNeckConv, self).__init__()
        self.nInput = nInput
        self.nOutput = nOutput
        self.kerSize = kerSize
        self.stride = stride
        self.pad = pad

        self.conv = nn.Conv2d(self.nInput, self.nOutput, kernel_size=self.kerSize, stride=self.stride, padding=self.pad,
                              bias=True)


    def forward(self, x):
        x = self.conv(x)
        return x


class MUNET(nn.Module):
    def __init__(self, D_in=3, D_out=2):
        super(MUNET, self).__init__()
        self.D_in = D_in
        self.D_out = D_out
        # ====================== STAGE 1 ===========================
        self.conv1_1 = basicConv(self.D_in, 64, 3, 1, 1)
        self.conv1_2 = basicConv(64, 64, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=False)
        # ====================== STAGE 2 ===========================
        self.conv2_1 = basicConv(64, 128, 3, 1, 1)
        self.conv2_2 = basicConv(128, 128, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2, 2, ceil_mode=False)
        # ====================== STAGE 3 ===========================
        self.conv3_1 = basicConv(128, 256, 3, 1, 1)
        self.conv3_2 = basicConv(256, 256, 3, 1, 1)
        self.conv3_3 = basicConv(256, 256, 3, 1, 1)
        self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=False)
        # ====================== STAGE 4 ===========================
        self.conv4_1 = basicConv(256, 512, 3, 1, 1)
        self.conv4_2 = basicConv(512, 512, 3, 1, 1)
        self.conv4_3 = basicConv(512, 512, 3, 1, 1)
        self.pool4 = nn.MaxPool2d(2, 2, ceil_mode=False)
        # ====================== STAGE 5 ===========================
        self.conv5_1 = basicConv(512, 512, 3, 1, 1)
        self.conv5_2 = basicConv(512, 512, 3, 1, 1)
        self.conv5_3 = basicConv(512, 512, 3, 1, 1)
        # ================== Pyramid Pooling =======================
        self.spp_pool2 = nn.AvgPool2d(16, 16)
        self.spp_pool2_conv = basicConv(512, 512, 1, 1, 0)

        self.spp_pool3 = nn.AvgPool2d(8, 8)
        self.spp_pool3_conv = basicConv(512, 512, 1, 1, 0)

        self.spp_pool6 = nn.AvgPool2d(4, 4)
        self.spp_pool6_conv = basicConv(512, 512, 1, 1, 0)

        # # ================ After Pyramid Pooling ===================
        # self.conv6_1 = basicConv(512*4, 512*2, 3, 1, 1)
        # self.conv6_2 = basicConv(512*2, 512, 3, 1, 1)

        self.conv6_1 = basicConv(512*4, 512, 3, 1, 1)
        # =================== Concat Pool4 =========================
        self.conv7_1 = basicConv(512+512, 256, 3, 1, 1)
        # =================== Concat Pool3 =========================
        self.conv8_1 = basicConv(256+256, 128, 3, 1, 1)
        # =================== Concat Pool2 =========================
        self.conv9_1 = basicConv(128+128, 64, 3, 1, 1)
        # =================== Concat Pool1 =========================
        self.conv10_1 = basicConv(64+64, 64, 3, 1, 1)
        # =================== Ouput Conv ===========================
        self.conv_end = bottleNeckConv(64, D_out)

    def forward(self, x):
        x, _ = self.conv1_1(x)
        x, _ = self.conv1_2(x)
        part1 = x
        x = self.pool1(x)

        x, _ = self.conv2_1(x)
        x, _ = self.conv2_2(x)
        part2 = x
        x = self.pool2(x)

        x, _ = self.conv3_1(x)
        x, _ = self.conv3_2(x)
        x, _ = self.conv3_3(x)
        part3 = x
        x = self.pool3(x)

        x, _ = self.conv4_1(x)
        x, _ = self.conv4_2(x)
        x, _ = self.conv4_3(x)
        part4 = x
        x = self.pool4(x)

        x, _ = self.conv5_1(x)
        x, _ = self.conv5_2(x)
        x, _ = self.conv5_3(x)

        flow0, flow1, flow2, flow3 = x, x, x, x       
        
        flow1 = self.spp_pool2(flow1)
        flow1, _ = self.spp_pool2_conv(flow1)
        flow1 = F.interpolate(flow1, size=[32, 32], mode='bilinear', align_corners=False)
        
        flow2 = self.spp_pool3(flow2)
        flow2, _ = self.spp_pool3_conv(flow2)
        flow2 = F.interpolate(flow2, size=[32, 32], mode='bilinear', align_corners=False)

        flow3 = self.spp_pool6(flow3)
        flow3, _ = self.spp_pool6_conv(flow3)
        flow3 = F.interpolate(flow3, size=[32, 32], mode='bilinear', align_corners=False)
        
        x = torch.cat((flow0, flow1, flow2, flow3), 1)
        x, _ = self.conv6_1(x)
        # x, _ = self.conv6_2(x)

        x = F.interpolate(x, size=[64, 64], mode='bilinear', align_corners=False)
        x = torch.cat((part4, x), 1)
        x, _ = self.conv7_1(x)

        x = F.interpolate(x, size=[128, 128], mode='bilinear', align_corners=False)
        x = torch.cat((part3, x), 1)
        x, _ = self.conv8_1(x)

        x = F.interpolate(x, size=[256, 256], mode='bilinear', align_corners=False)
        x = torch.cat((part2, x), 1)
        x, _ = self.conv9_1(x)

        x = F.interpolate(x, size=[512, 512], mode='bilinear', align_corners=False)
        x = torch.cat((part1, x), 1)
        x, _ = self.conv10_1(x)

        x = self.conv_end(x)
        return x

    def getParamsPretrained(self, params_vgg16):
        pp_dict = self.state_dict()
        f_keys_vgg16 = open('VGG16-param-keys.txt')
        keys_vgg16 = []
        for line in f_keys_vgg16.readlines():
            line = line.strip()
            keys_vgg16.append(line)
        f_keys_vgg16.close()

        for i, key in enumerate(pp_dict.keys()):
            print(str(keys_vgg16[i] + ' done!\n'))
            pp_dict[key] = params_vgg16[keys_vgg16[i]]
            i+=1
            if i==91:
                break
        self.load_state_dict(pp_dict)

    def fixPretrianedParams(self, switch=False):
        if switch:
            for i, m in enumerate(self.modules()):
                print
                if isinstance(m, basicConv):
                    continue
                elif isinstance(m, nn.Conv2d):
                    m.weight.requires_grad = False
                elif isinstance(m, nn.BatchNorm2d):
                    m.eval()
                if i == 43:
                    break



    # def FeatureMapExtractor(self, input):
    #     x, conv1_1_out = self.conv1_1(input)
    #     x, conv1_2_out = self.conv1_2(x)
    #     x = self.pool1(x)
    #
    #     part1, pool1_out = x, x
    #
    #     x, conv2_1_out = self.conv2_1(x)
    #     x, conv2_2_out = self.conv2_2(x)
    #     x = self.pool2(x)
    #     part2, pool2_out = x, x
    #
    #     x, conv3_1_out = self.conv3_1(x)
    #     x, conv3_2_out = self.conv3_2(x)
    #     x, conv3_3_out = self.conv3_3(x)
    #     x = self.pool3(x)
    #     part3, pool3_out = x, x
    #
    #     x, conv4_1_out = self.conv4_1(x)
    #     x, conv4_2_out = self.conv4_2(x)
    #     x, conv4_3_out = self.conv4_3(x)
    #
    #     x, conv5_1_out = self.conv5_1(x)
    #     x, conv5_2_out = self.conv5_2(x)
    #     x, conv5_3_out = self.conv5_3(x)
    #
    #     x, fc_out = self.fc(x)
    #     flow0, flow1, flow2, flow3 = x, x, x, x
    #
    #     x = self.spp_pool2(flow1)
    #     spp_pool2_out = x
    #     x, spp_pool2_conv_out = self.spp_pool2_conv(x)
    #     x = F.interpolate(x, size=[60, 60], mode='bilinear', align_corners=False)
    #     spp_pool2_interp_out = x
    #
    #     x = self.spp_pool3(flow2)
    #     spp_pool3_out = x
    #     x, spp_pool3_conv_out = self.spp_pool3_conv(x)
    #     x = F.interpolate(x, size=[60, 60], mode='bilinear', align_corners=False)
    #     spp_pool3_interp_out = x
    #
    #     x = self.spp_pool6(flow3)
    #     spp_pool6_out = x
    #     x, spp_pool6_conv_out = self.spp_pool6_conv(x)
    #     x = F.interpolate(x, size=[60, 60], mode='bilinear', align_corners=False)
    #     spp_pool6_interp_out = x
    #
    #     x = torch.cat((flow0, spp_pool2_interp_out, spp_pool3_interp_out, spp_pool6_interp_out), 1)
    #     concat_1_out = x
    #
    #     x, conv6_1_out = self.conv6_1(x)
    #     x, conv6_2_out = self.conv6_2(x)
    #
    #     x = torch.cat((part3, x), 1)
    #     concat_2_out = x
    #     x, conv6_3_out = self.conv6_3(x)
    #     x = F.interpolate(x, size=[119, 119], mode='bilinear', align_corners=False)
    #     interp_2_out = x
    #
    #     x = torch.cat((part2, x), 1)
    #     concat_3_out = x
    #     x, conv7_1_out = self.conv7_1(x)
    #     x, conv7_2_out = self.conv7_2(x)
    #     x = F.interpolate(x, size=[237, 237], mode='bilinear', align_corners=False)
    #     interp_3_out = x
    #
    #     x = torch.cat((part1, x), 1)
    #     concat_4_out = x
    #     x, conv8_1_out = self.conv8_1(x)
    #     x, conv8_2_out = self.conv8_2(x)
    #     x = self.conv8_3(x)
    #     conv8_3_out = x
    #     output = F.interpolate(x, [473, 473], mode='bilinear', align_corners=False)
    #
    #     key_names = ['conv1_1_out', 'conv1_2_out', 'pool1_out',
    #         'conv2_1_out', 'conv2_2_out', 'pool2_out',
    #         'conv3_1_out', 'conv3_2_out', 'conv3_3_out', 'pool3_out',
    #         'conv4_1_out', 'conv4_2_out', 'conv4_3_out',
    #         'conv5_1_out', 'conv5_2_out', 'conv5_3_out',
    #         'fc_out',
    #         'spp_pool2_out', 'spp_pool2_conv_out', 'spp_pool2_interp_out',
    #         'spp_pool3_out', 'spp_pool3_conv_out', 'spp_pool3_interp_out',
    #         'spp_pool6_out', 'spp_pool6_conv_out', 'spp_pool6_interp_out',
    #         'concat_1_out',
    #         'conv6_1_out', 'conv6_2_out',
    #         'concat_2_out',
    #         'conv6_3_out', 'interp_2_out',
    #         'concat_3_out',
    #         'conv7_1_out', 'conv7_2_out', 'interp_3_out',
    #         'concat_4_out',
    #         'conv8_1_out', 'conv8_2_out', 'conv8_3_out']
    #
    #     key_values = [conv1_1_out, conv1_2_out, pool1_out,
    #         conv2_1_out, conv2_2_out, pool2_out,
    #         conv3_1_out, conv3_2_out, conv3_3_out, pool3_out,
    #         conv4_1_out, conv4_2_out, conv4_3_out,
    #         conv5_1_out, conv5_2_out, conv5_3_out,
    #         fc_out,
    #         spp_pool2_out, spp_pool2_conv_out, spp_pool2_interp_out,
    #         spp_pool3_out, spp_pool3_conv_out, spp_pool3_interp_out,
    #         spp_pool6_out, spp_pool6_conv_out, spp_pool6_interp_out,
    #         concat_1_out,
    #         conv6_1_out, conv6_2_out,
    #         concat_2_out,
    #         conv6_3_out, interp_2_out,
    #         concat_3_out,
    #         conv7_1_out, conv7_2_out, interp_3_out,
    #         concat_4_out,
    #         conv8_1_out, conv8_2_out, conv8_3_out]
    #
    #     feaMaps = {}
    #     for i, name in enumerate(key_names):
    #         feaMaps[name] = key_values[i]
    #     return output, feaMaps


