from builder import reverse, Asset, TwoWayAsset

a = Asset(1.875, [5], 1.875, [3, 2])
print(TwoWayAsset(a, a))